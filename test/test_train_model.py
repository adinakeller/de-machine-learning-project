import pandas as pd
from pandas.testing import assert_frame_equal
from model.train_model import load_csv, save_trained_model
from sklearn.linear_model import LogisticRegression

def test_loads_csv_file(tmp_path):
    file = tmp_path / 'test.csv'
    test_df = pd.DataFrame({'text': ['im on a walk'], 'label': [1]})

    test_df.to_csv(file, index=False)
    result = load_csv(file)
    print(result.dtypes)
    print(test_df.dtypes)

    assert_frame_equal(result, test_df)
    assert result['text'].dtypes == test_df['text'].dtypes
    assert result['label'].dtypes == test_df['label'].dtypes

def test_saves_model_pkl_file(tmp_path):
    model = LogisticRegression()
    file_name = tmp_path / 'cleaned.csv'
    save_trained_model(model, file_name)

    assert file_name.exists()


