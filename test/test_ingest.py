from model.ingest import load, save_to_csv

def test_load_dataset(mocker):
    mock_dataset = {
        'train': [
            {'text': 'im going on a run', 'label': 1},
            {'text': 'im happy', 'label': 3}
            ]
        }
    
    mocker.patch('ingest.load_dataset', return_value=mock_dataset)
    result = load(mock_dataset, 'unsplit')

    assert len(result) == 2
    assert result == [{'text': 'im going on a run', 'label': 1}, {'text': 'im happy', 'label': 3}]

def test_saves_csv_to_correct_file():
    data = [
        {'text': 'im going on a run', 'label': 1},
        {'text': 'im happy', 'label': 3}
        ]
    
    file_name = 'cleaned.csv'

    result = save_to_csv(file_name, data)

    assert len(result) == 2
    assert list(result.columns) == ['text', 'label']
    assert result.loc[0, 'text'] == 'im going on a run'
    assert result.loc[1, 'label'] == 3

