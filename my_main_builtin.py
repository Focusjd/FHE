# Import decision tree classifier method.
from sklearn.tree import DecisionTreeClassifier

# For displaying the tree.
# from sklearn.externals.six import StringIO
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz, export_text
from sklearn.tree import *
import pydotplus
import time

# For oversampling the imbalanced datasets
from imblearn.over_sampling import RandomOverSampler


# Load the training data set based on file name
def load_dataset(fname):
    with open(fname) as f:
        content = f.readlines()

    # Remove newline characters
    content = [x.strip('\n') for x in content]

    # Split each data vector into a sub-list
    content = [ x.split(' ') for x in content]

    # Remove the ID 
    # content = [ x[1:11] for x in content]

    # Convert to real numbers
    content = [ [float(item) for item in sublist] for sublist in content]

    # Split up the datapoints and their classes
    datapoints = [ sublist[0:len(sublist)-1] for sublist in content ]
    classes = [sublist[-1] for sublist in content]
    classes = [str(item) for item in classes]
    
    # Return. 
    return datapoints, classes

def main():
    print_test = ''
    print_train = ''

    # Optimal depth is 5.
    tree_depth = 5

    

    # for sbgrp in range(1,11):
    for sbgrp in range(1,2):
        # Storage. 
        TP = TN = FP = FN = 0
        TPt = TNt = FPt = FNt = 0
        # Load the dataset.
        # fin = "train" + str(sbgrp) + ".txt"
        fin = "./tox21_data/ranged_data/train_labeled.txt"
        data, target = load_dataset(fin)

        # Oversample
        ros = RandomOverSampler(random_state=0)
        # data_resamp, target_resamp = ros.fit_sample(data, target)
        data_resamp, target_resamp = ros.fit_resample(data, target)

        # Initialize the classifier.
        clf = DecisionTreeClassifier(random_state=0, max_depth=tree_depth,
                                     min_samples_split=3,
                                     min_samples_leaf=3)

        # Fit the classifier.
        #clf = clf.fit(data, target)
        clf = clf.fit(data_resamp, target_resamp)

        # Save dot file
        # f_dot = "/Users/dianjiao/FHE_project/FHECode/Files_DTBreastCancer/tree" + str(sbgrp) + ".dot"
        f_dot = "./tox21_data/ranged_data/training_tree_tox21_ranged" + ".dot"

        dotfile = open(f_dot, 'w')
        export_graphviz(clf, out_file = dotfile, class_names = clf.classes_)
        dotfile.close()

        # txt_file = "tree_structure.txt"
        # with open(txt_file, 'w') as outfile:
        #     tree = export_text(clf)
        #     outfile.write(tree)
        # print(tree)


        # Display the tree
        dot_data = StringIO()
        export_graphviz(clf, out_file=dot_data, filled=True,
                        rounded=True, special_characters=True,
                        class_names = clf.classes_)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        Image(graph.create_png())
        f_pdf = "./tox21_data/ranged_data/training_tree_tox21_ranged_clfpdf.pdf"
        graph.write_pdf(f_pdf)

        # Check performance on training data set.
        time_start = time.perf_counter()
        predicted_classes = clf.predict(data)
        total_timet = (time.perf_counter() - time_start)
        for i in range(len(predicted_classes)):
            if (predicted_classes[i]=='2.0') & (target[i]=='2.0'):
                TPt += 1
            elif (predicted_classes[i]=='2.0') & (target[i]=='4.0'):
                FPt += 1
            elif (predicted_classes[i]=='4.0') & (target[i]=='4.0'):
                TNt += 1
            else:
                FNt += 1
                
        # Validate performance on test data set. 
        # ftest = "test" + str(sbgrp) + ".txt"
        ftest = './tox21_data/ranged_data/test_labeled.txt'

        data_test, true_classes = load_dataset(ftest)

        time_start = time.perf_counter()
        predicted_classes = clf.predict(data_test)
        total_time = (time.perf_counter()-time_start)
        for i in range(len(predicted_classes)):
            if (predicted_classes[i]=='2.0') & (true_classes[i]=='2.0'):
                TP += 1
            elif (predicted_classes[i]=='2.0') & (true_classes[i]=='4.0'):
                FP += 1
            elif (predicted_classes[i]=='4.0') & (true_classes[i]=='4.0'):
                TN += 1
            else:
                FN += 1

        print_train += str(sbgrp) + ',' + str(TPt) + ',' + str(TNt) + ',' + str(FPt) + ','+ str(FNt) + str(total_timet) + '\n'
        print_test += str(sbgrp) + ',' + str(TP) + ',' + str(TN) + ',' + str(FP) + ','+ str(FN) + str(total_time)+"\n"
        #print_train += str(total_timet) + '\n'
        #print_test += str(total_time)+"\n"
    #print(
    print(print_train)
    print(print_test)

main()
