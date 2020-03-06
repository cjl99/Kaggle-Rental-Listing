feature_extraction.py for extracting the train features
feature_extraction.py need train2.json as input
feature_extraction.py output trainxxx.json
train2.json is generated from outlier.py

test_extraction.py for extracting the test features
test_extraction.py need trainxxx.json train2.json and test.json as input
test_extraction.py output new_test.json

fisher_score for extracting features which have high fisher score
test_extraction.py need trainxxx.json as input
test_extraction.py it output train4.json

logistic_regression.py for logistic regression model
decision_tree for decisiom_tree model

confusion.py for generating the figure of confusion matrix

result predication:
lr_submission.csv
submission_tree.csv