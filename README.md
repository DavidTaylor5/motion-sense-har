# motion-sense-har
Har classification using CNN and logistic regression


12 Features, 6 Classes

Sensitive attributes are kept in the file 'data_subjects_info.csv'

I try to get data in the format [X_train, y_train, X_test, y_test]. Each participant should have an array of these values, split 80/20.
Pooled Data is in the same format.
CNN uses images of 50 timesteps (50, 12)
CNN uses one shot vectors for y values.

