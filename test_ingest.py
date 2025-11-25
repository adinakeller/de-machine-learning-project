from ingest import load_dataset, save_to_csv

def test_load_dataset(mocker):
    mock_dataset = {
        'train': {'text': 'im going on a run', 'label': 1},
        'test': {'text': 'im happy', 'label': 3}
        }
    
    mocker.patch('ingest.load_dataset', return_value=mock_dataset)
    result = load_dataset('dataset', 'split')

    assert result == {'text': 'im going on a run', 'label': 1}

def test_saves_csv_to_correct_file():
    data = [
        {'text': 'im going on a run', 'label': 1},
        {'text': 'im happy', 'label': 3}
        ]
    
    file_name = 'cleaned.csv'

    result = save_to_csv(file_name, data)

    assert result == file_name


