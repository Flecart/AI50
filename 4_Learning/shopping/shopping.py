import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py shopping.csv")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    def date_parser(date):
        """
        This function parses string months into number codes. If the string is 
        unknown, it raises an error
        """
        if date == "Jan":
            return 0
        elif date == "Feb":
            return 1
        elif date == "Mar":
            return 2
        elif date == "May":
            return 3
        elif date == "Apr":
            return 4
        elif date == "June":
            return 5
        elif date == "Jul":
            return 6
        elif date == "Aug":
            return 7
        elif date == "Sep":
            return 8
        elif date == "Oct":
            return 9
        elif date == "Nov":
            return 10
        elif date == "Dec":
            return 10
        else:
            raise KeyError(f"Unknown string: got { date }")

    with open(filename) as f:
        reader = csv.reader(f)

        # getting rid of the headers
        next(reader)
        evidence_list = []
        labels = []
        for row in reader:
            administrative = [int(row[0])]

            # maybe i should try catch every value conversion?
            try:
                administrative_duration = [float(row[1])]
            except ValueError:
                print(row[1], "error at administrative duration")
                sys.exit(1)

            informational = [int(row[2])]
            informational_duration = [float(row[3])]
            productRelated = [int(row[4])]
            productRelated_duration = [float(row[5])]
            bounceRates = [float(row[6])]
            exitRates = [float(row[7])]
            pageValues = [float(row[8])]
            specialDay = [float(row[9])]
            month = [date_parser(row[10])]
            operatingSystems = [int(row[11])]
            browser = [int(row[12])]
            region = [int(row[13])]
            trafficType = [int(row[14])]
            visitorType = [1 if row[15] == "Returning_Visitor" else 0]
            weekend = [0 if row[16] == "TRUE" else 0]
            evidence = administrative + administrative_duration + informational + informational_duration + productRelated + \
                productRelated_duration + bounceRates + exitRates + pageValues + specialDay + month + operatingSystems + \
                browser + region + trafficType + visitorType + weekend
            
            evidence_list.append(evidence)
            labels.append(1 if row[17] == "TRUE" else 0)
            
        return (evidence_list, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # this is the same process i found in the notes at
    # https://cs50.harvard.edu/ai/2020/notes/4/

    # creating the classifier
    model = KNeighborsClassifier(n_neighbors=1)
    # fitting the data to the model
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    positive_correct = 0
    positive_incorrect = 0
    negative_correct = 0
    negative_incorrect = 0
    for label, prediction in zip(labels, predictions):

        # sensitivity
        if label == 1:
            if prediction == label:
                positive_correct += 1
            else:
                positive_incorrect += 1

        # specificity
        else:
            if prediction == label:
                negative_correct += 1
            else:
                negative_incorrect += 1

    return (positive_correct / (positive_correct + positive_incorrect), negative_correct / (negative_correct + negative_incorrect))



if __name__ == "__main__":
    main()
