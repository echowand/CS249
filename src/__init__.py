import pandas as pa
import numpy as np
from sklearn.cross_validation import KFold

promoted_content = pa.read_csv('../resources/promoted_content.csv', header=0)
events = pa.read_csv('../resources/events.csv', header=0)

data = events.set_index('document_id').join(promoted_content.set_index('document_id'))
print data

data = data[np.isfinite(data['ad_id'])]
data_selected = data.ix[:,[0,1,2,6]]
data_selected.to_csv("promote_event_sel.csv", sep=',')
# split the data into 10 folds
# f10 = KFold(len(data), n_folds=10, shuffle=True, random_state=None)


