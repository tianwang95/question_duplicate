package ai.eloquent.qa;

import edu.stanford.nlp.classify.Classifier;
import edu.stanford.nlp.classify.ClassifierFactory;
import edu.stanford.nlp.classify.LinearClassifierFactory;
import edu.stanford.nlp.classify.RVFDataset;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.pipeline.CoreNLPProtos;
import edu.stanford.nlp.simple.Document;
import edu.stanford.nlp.simple.Sentence;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Pair;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * TODO(gabor) JavaDoc
 *
 * @author <a href="mailto:gabor@eloquent.ai">Gabor Angeli</a>
 */
public class GaborsSandbox {


  private static class Datum {
    public final Sentence a;
    public final Sentence b;
    public final boolean  synonym;


    /**
     * An alignment from each conclusion token to a premise token
     */
    public final int[] conclusionToPremiseAlignment;

    public Datum(Sentence a, Sentence b, boolean synonym) {
      this.a = a;
      this.b = b;
      this.synonym = synonym;
      this.conclusionToPremiseAlignment = new int[b.length()];
    }


    public void bagOfWords(Counter<String> features) {
      a.words().forEach(features::incrementCount);
      b.words().forEach(features::incrementCount);
    }


    public Counter<String> featurize() {
      Counter<String> features = new ClassicCounter<>();
      bagOfWords(features);
      return features;
    }


    /**
     * All phrase table candidates from this alignment pair.
     *
     * @param maxLength The maximum length of a phrase.
     *
     * @return A collection of Span pairs for every pair of spans to add to the phrase table.
     */
    @SuppressWarnings("UnnecessaryLabelOnContinueStatement")
    private List<Pair<Span, Span>> candidatePhraseSpans(int maxLength) {
      // Compute the a -> conclusion alignments
      List<Set<Integer>> aToConclusionAlignment = new ArrayList<>(a.length());
      for (int i = 0; i < a.length(); ++i) {
        aToConclusionAlignment.add(new HashSet<>());
      }
      for (int i = 0; i < b.length(); ++i) {
        if (conclusionToPremiseAlignment[i] >= 0) {
          aToConclusionAlignment.get(conclusionToPremiseAlignment[i]).add(i);
        }
      }

      // Compute the phrases
      List<Pair<Span, Span>> phrases = new ArrayList<>(32);
      for (int length = 2; length <= maxLength; ++length) {
        NEXT_SPAN: for (int start = 0; start <= b.length() - length; ++start) {
          int end = start + length;
          assert end <= b.length();

          // Compute the initial candidate span
          int min = 999;    // a start candidate
          int max = -999;   // a end candidate
          for (int i = start; i < end; ++i) {
            if (conclusionToPremiseAlignment[i] >= 0) {
              min = conclusionToPremiseAlignment[i] < min ? conclusionToPremiseAlignment[i] : min;
              max = conclusionToPremiseAlignment[i] >= max ? conclusionToPremiseAlignment[i] + 1 : max;
            }
          }
          if (min < 0 || max > conclusionToPremiseAlignment.length ||
              min >= conclusionToPremiseAlignment.length || max <= 0) {
            continue NEXT_SPAN;
          }

          // Check candidate span viability
          for (int aI = min; aI < max; ++aI) {
            Set<Integer> conclusionsForPremise = aToConclusionAlignment.get(aI);
            for (int conclusionI : conclusionsForPremise) {
              if (conclusionI < start || conclusionI >= end) {
                continue NEXT_SPAN;  // Alignment is not viable
              }
            }
          }

          // Try to extend the alignment over null tokens while possible
          int extremeMin = min;
          int extremeMax = max;
          while (extremeMin > 0 && aToConclusionAlignment.get(extremeMin - 1).isEmpty()) {
            extremeMin -= 1;
          }
          while (extremeMax < aToConclusionAlignment.size() && aToConclusionAlignment.get(extremeMax).isEmpty()) {
            extremeMax += 1;
          }

          // Add the alignment
          for (int aStart = extremeMin; aStart <= min; ++aStart) {
            for (int aEnd = extremeMax; aEnd >= max; --aEnd) {
              phrases.add(Pair.makePair(new Span(aStart, aEnd), new Span(start, end)));
            }
          }
        }
      }

      return phrases;
    }


    public QAProtos.QADatum serialize() {
      return QAProtos.QADatum.newBuilder()
          .setA(a.serialize())
          .setB(b.serialize())
          .setLabel(this.synonym)
          .build();
    }


    public static List<Datum> deserialize(QAProtos.QADataset dataset) {
      return dataset.getDatumList().stream()
          .map(x -> new Datum(
              new Sentence(x.getA()),
              new Sentence(x.getB()),
              x.getLabel()
          ))
          .collect(Collectors.toList());
    }
  }


  private static Document read(InputStream reader) throws IOException {
    int ch1 = reader.read();
    int ch2 = reader.read();
    int ch3 = reader.read();
    int ch4 = reader.read();
    if ((ch1 | ch2 | ch3 | ch4) < 0) {
      return null;
    }
    int size = ((ch1) + (ch2 << 8) + (ch3 << 16) + (ch4 << 24));
    byte[] protoData = new byte[size];
    reader.read(protoData, 0, size);
    CoreNLPProtos.Document proto = CoreNLPProtos.Document.parseFrom(protoData);
    Document doc = new Document(proto);
    return doc;
  }


  public static void main(String[] args) throws IOException, ClassNotFoundException {
    int trainCount = 10000;
    int testCount  = 10000;


    forceTrack("Loading data");
    InputStream reader = IOUtils.getInputStreamFromURLOrClasspathOrFileSystem("quora_annotated.data.gz");
    Iterator<Boolean> labels = Stream.of(IOUtils.slurpFile("labels.txt").split("\n")).map(x -> Objects.equals(x, "1")).iterator();
    int numread = 0;

    RVFDataset<Boolean, String> train = new RVFDataset<>();
    RVFDataset<Boolean, String> test = new RVFDataset<>();
    while (labels.hasNext()) {
      Document a = read(reader);
      Document b = read(reader);
      if (a.sentences().size() != 0 && b.sentences().size() != 0) {
        if ((++numread % 1000) == 0) {
          log("Read " + numread + " examples: " + (Runtime.getRuntime().totalMemory() / (1024 * 1024)) + "M");
        }
        Datum datum = new Datum(a.sentence(0), b.sentence(0), labels.next());
        if (numread > trainCount + testCount) {
          break;
        } else if (numread > trainCount) {
          test.add(new RVFDatum<>(datum.featurize(), datum.synonym));
        } else {
          train.add(new RVFDatum<>(datum.featurize(), datum.synonym));

        }
      }
    }
    endTrack("Loading data");


    forceTrack("Training classifier");
    ClassifierFactory<Boolean, String, Classifier<Boolean, String>> factory = new LinearClassifierFactory<>();
    Classifier<Boolean, String> classifier = factory.trainClassifier(train);
    endTrack("Training classifier");

    forceTrack("Evaluating");
    double accuracy = classifier.evaluateAccuracy(test);
    log("Accuracy: " + new DecimalFormat("0.000%").format(accuracy));
    endTrack("Evaluating");


    FileOutputStream os = new FileOutputStream("dataset.ser");
    os.close();
  }
}
