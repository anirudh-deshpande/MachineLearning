"""
SMS messages retrieved from http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/
"""

import pandas as pd

def get_messages_df(file_name):
    messages_df = pd.read_table(file_name,
                                sep='\t',
                                header=None,
                                names=['label', 'message'])
    '''
    To evaluate performance using scikit-learn, we convert 
        catagorical labels 'ham' and 'spam' into integers [0, 1]
    '''
    messages_df['label'] = messages_df.label.map({'ham': 0, 'spam': 1})

    # Top and bottom 10 messages
    # print messages_df.head(10)
    # print messages_df.tail(10)

    # Size of dataset (rows, columns)
    # print messages_df.shape
    return messages_df

