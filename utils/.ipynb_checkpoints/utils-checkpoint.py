import pandas as pd
import os

def load_data(truncate_transcripts=True) -> pd.DataFrame:
    data_path = './data/'
    question_df = pd.read_csv(os.path.join(data_path, 'acquired-qa-evaluation.csv'), encoding='unicode_escape')
    metadata_df = pd.read_csv(os.path.join(data_path, 'acquired_metadata.csv'))
    transcripts = load_transcripts()
    metadata_df = metadata_df[metadata_df['has_transcript']]
    metadata_df['transcript'] = metadata_df['file_name'].apply(lambda x: transcripts[x])
    question_df = question_df[['question', 'human_answer', 'file_name']]
    df = pd.merge(metadata_df, question_df, on='file_name', how='inner')
    # token limit
    if truncate_transcripts:
        df['transcript'] = df['transcript'].apply(lambda x: ' '.join([token for token in x.split(' ')][0:10000]))
    return df

def load_transcripts() -> dict:
    data_path = './data/'
    transcripts = {}
    for file in os.listdir(os.path.join(data_path, 'acquired-individual-transcripts/acquired-individual-transcripts/')):
        fname = file.split('.')[0]
        transcripts[fname] = open(os.path.join(data_path, 'acquired-individual-transcripts/acquired-individual-transcripts/', file)).read()
    return transcripts
    