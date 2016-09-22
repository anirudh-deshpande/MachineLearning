import NaiveBayes
import KNearestNeighbours

if __name__ == "__main__":
    NaiveBayes.print_naivebayes_accuracy('train.txt', 'test.txt')
    KNearestNeighbours.print_knn_accuracy('train.txt', 'test.txt')