print("Making imports...")
from TrainTestNaiveBayes import trainTestNaiveBayes
from TestDecisionTrees import testDecisionTrees
from TestMLPs import testMLPs
from TestCNNs import testCNNs

# Run this file to test all the saved models

def testAll():
    accuracies = []
    precisions = []
    recalls = []
    f1Measures = []

    print("\n\nTesting Naive Bayes...")
    a, p, r, f = trainTestNaiveBayes()
    accuracies.extend(a)
    precisions.extend(p)
    recalls.extend(r)
    f1Measures.extend(f)
    
    print("\n\nTesting Decision Trees...")
    a, p, r, f = testDecisionTrees()
    accuracies.extend(a)
    precisions.extend(p)
    recalls.extend(r)
    f1Measures.extend(f)

    print("\n\nTesting MLPs...")
    a, p, r, f = testMLPs()
    accuracies.extend(a)
    precisions.extend(p)
    recalls.extend(r)
    f1Measures.extend(f)

    print("\n\nTesting CNNs...")
    a, p, r, f = testCNNs()
    accuracies.extend(a)
    precisions.extend(p)
    recalls.extend(r)
    f1Measures.extend(f)

    modelNames = [
        "Custom NB Classifier",
        "Scikit NB Classifier",
        "Custom DT MaxDepth 10",
        "Custom DT MaxDepth 20",
        "Custom DT MaxDepth 50",
        "Scikit DT MaxDepth 10",
        "Scikit DT MaxDepth 20",
        "Scikit DT MaxDepth 50",
        "Normal MLP",
        "Deepened MLP",
        "Narrowed MLP",
        "Widened MLP",
        "Normal CNN",
        "Shallowed CNN",
        "Smaller Kernel CNN",
        "Larger Kernel CNN"
    ]

    print("\nEvaluation Table: ")
    row_format = "{:<25} {:<10} {:<10} {:<10} {:<10}"
    print(row_format.format("Model:", "Accuracy:", "Precision:", "Recall:", "F1:"))
    for i in range(16):
        rowStr = row_format.format(modelNames[i], f"{accuracies[i]:.4f}", f"{precisions[i]:.4f}", f"{recalls[i]:.4f}", f"{f1Measures[i]:.4f}")
        print(rowStr)
    print("")

if __name__ == "__main__":
    testAll()