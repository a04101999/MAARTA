import os
import polars as pl
import json
import random
from polars import DataFrame
import time

"""
This script generates a dataset with perceptual errors using the EGD-CXR dataset. 
https://physionet.org/content/egd-cxr/1.0.0/

"""


def contains_valid_timestamps(transcript):

    if len(transcript['time_stamped_text']) == 0:
        return False

    size = len(transcript['time_stamped_text'])
    if transcript['time_stamped_text'][size - 1]['phrase'][-1] != ".":
        transcript['time_stamped_text'][size - 1]['phrase'] += "."

    for ts in transcript['time_stamped_text']:
        num_periods = ts['phrase'].count('.')
        if num_periods > 1:
            return False

    return True


def label_sentences(sentences: list[str], timestamps: list[dict]):
    speaking_durations = []  # speaking_durations[i] corresponds to sentences[i]

    tsIndx = 0
    for s in sentences:
        words = s.split(" ")
        ending_word = words[-1]

        begin_time = timestamps[tsIndx]['begin_time']

        while tsIndx < len(timestamps) and ending_word not in timestamps[tsIndx]['phrase']:
            tsIndx += 1

        if ending_word in timestamps[tsIndx]['phrase']:
            end_time = timestamps[tsIndx]['end_time']
            speaking_durations.append(
                {'sentence': s, 'begin_time': begin_time, 'end_time': end_time}
            )
            tsIndx += 1

    return speaking_durations


def label_abnormality_transcript_with_timestamps(transcript):

    timestamps = transcript['time_stamped_text']
    # remove leading and trailing white space
    full_text = transcript['full_text'].strip().lower()

    sentences = full_text.split(". ")  # split into individual sentences

    for i, s in enumerate(sentences):
        if s[-1] != ".":
            sentences[i] = s + "."

    sentence_with_timestamps = label_sentences(sentences, timestamps)

    abnormality_sentences_with_timestamps = []

    for swt in sentence_with_timestamps:
        # remove trailing period
        sentence = swt['sentence'][:len(swt['sentence']) - 1]
        words = set(sentence.split(" "))
        if "no" not in words and "not" not in words:
            abnormality_sentences_with_timestamps.append(swt)

    return abnormality_sentences_with_timestamps


def create_class1_perceptual_error(csv_path: str, removed_sentence_indx: int, abnormality_sentences_with_timestamps: list[dict]):
    """
    Missed abnormality due to removed transcription sentence
    """
    initial_fixations_df = pl.read_csv(os.path.join(csv_path, "fixations.csv"))

    removed_sentence_begin_time = abnormality_sentences_with_timestamps[
        removed_sentence_indx]['begin_time']
    removed_sentence_end_time = abnormality_sentences_with_timestamps[
        removed_sentence_indx]['end_time']

    missed_fixations_df = initial_fixations_df.filter((pl.col('Time (in secs)') >= removed_sentence_begin_time) & (
        pl.col('Time (in secs)') <= removed_sentence_end_time))

    remaining_fixations_df = initial_fixations_df.filter(~((pl.col('Time (in secs)') >= removed_sentence_begin_time) & (
        pl.col('Time (in secs)') <= removed_sentence_end_time)))

    return missed_fixations_df, remaining_fixations_df


def half_eye_gaze_fixation(FPOGD):
    return FPOGD * 0.5


def create_class2_perceptual_error(csv_path: str, removed_sentence_indx: int, abnormality_sentences_with_timestamps: list[dict], fixation_reducer):
    """
    Missed abnormality due to reduced fixation duration
    """
    initial_fixations_df = pl.read_csv(os.path.join(csv_path, "fixations.csv"))

    removed_sentence_begin_time = abnormality_sentences_with_timestamps[
        removed_sentence_indx]['begin_time']
    removed_sentence_end_time = abnormality_sentences_with_timestamps[
        removed_sentence_indx]['end_time']

    fixations_reduced_df = initial_fixations_df.filter(
        (pl.col('Time (in secs)') >= removed_sentence_begin_time) &
        (pl.col('Time (in secs)') <= removed_sentence_end_time)
    )

    reduced_fixations_final_output_df = initial_fixations_df.with_columns(
        pl.when((pl.col('Time (in secs)') >= removed_sentence_begin_time)
                & (pl.col('Time (in secs)') <= removed_sentence_end_time))
        .then(fixation_reducer(pl.col('FPOGD')))
        .otherwise(pl.col('FPOGD'))
        .alias('FPOGD')
    )

    return fixations_reduced_df, reduced_fixations_final_output_df


def create_both_class1_and_class2_perceptual_error(csv_path: str, abnormality_sentences_with_timestamps: list[dict], fixation_reducer):
    """
    Missed abnormality due to removed transcription sentence and reduced fixation duration
    """

    c1_removed_sentence_indx = random.randint(
        0, len(abnormality_sentences_with_timestamps) - 1)
    c2_removed_sentence_indx = c1_removed_sentence_indx

    while c2_removed_sentence_indx == c1_removed_sentence_indx:
        c2_removed_sentence_indx = random.randint(
            0, len(abnormality_sentences_with_timestamps) - 1)

    initial_fixations_df = pl.read_csv(os.path.join(csv_path, "fixations.csv"))

    #! Add class 1 perceptual error
    c1_removed_sentence_begin_time = abnormality_sentences_with_timestamps[
        c1_removed_sentence_indx]['begin_time']
    c1_removed_sentence_end_time = abnormality_sentences_with_timestamps[
        c1_removed_sentence_indx]['end_time']

    missed_fixations_df = initial_fixations_df.filter((pl.col('Time (in secs)') >= c1_removed_sentence_begin_time) & (
        pl.col('Time (in secs)') <= c1_removed_sentence_end_time))

    remaining_fixations_df = initial_fixations_df.filter(~((pl.col('Time (in secs)') >= c1_removed_sentence_begin_time) & (
        pl.col('Time (in secs)') <= c1_removed_sentence_end_time)))

    #! Add class 2 perceptual error
    c2_removed_sentence_begin_time = abnormality_sentences_with_timestamps[
        c2_removed_sentence_indx]['begin_time']
    c2_removed_sentence_end_time = abnormality_sentences_with_timestamps[
        c2_removed_sentence_indx]['end_time']

    fixations_reduced_df = remaining_fixations_df.filter(
        (pl.col('Time (in secs)') >= c2_removed_sentence_begin_time) &
        (pl.col('Time (in secs)') <= c2_removed_sentence_end_time)
    )

    final_fixations_output_df = remaining_fixations_df.with_columns(
        pl.when((pl.col('Time (in secs)') >= c2_removed_sentence_begin_time)
                & (pl.col('Time (in secs)') <= c2_removed_sentence_end_time))
        .then(fixation_reducer(pl.col('FPOGD')))
        .otherwise(pl.col('FPOGD'))
        .alias('FPOGD')
    )

    return c1_removed_sentence_indx, c2_removed_sentence_indx, missed_fixations_df, remaining_fixations_df, fixations_reduced_df, final_fixations_output_df


def create_class3_perceptual_error(csv_path: str):
    """
    Missed abnormality due to less experience
    """
    return pl.read_csv(os.path.join(csv_path, "fixations.csv"))


def get_correct_data(csv_path: str):
    return pl.read_csv(os.path.join(csv_path, "fixations.csv"))


def convert_df_to_dict(df: DataFrame):
    # keys_to_extract = set(
    #     {'FPOGX', 'FPOGY', 'FPOGD', 'Time (in secs)'})
    keys_to_extract = set(
        {'X_ORIGINAL', 'Y_ORIGINAL', 'FPOGD', 'Time (in secs)'})
    df_dict = df.to_dict(as_series=False)
    return {key: df_dict[key] for key in keys_to_extract if key in df_dict}


def main():
    """
    Divide number of samples into 
    """
    start_time = time.time()

    perceptual_error_class_labels = {
        1: 'Missed abnormality due to lack of fixation on region of interest',
        2: 'Missed abnormality due to reduced fixation duration on region of interest',
        3: 'Missed abnormality due to less experience in detecting abnormalities',
    }

    fixation_fpath = "fixations"
    audio_seg_transcript_fpath = "audio_segmentation_transcripts"
    dicom_ids = os.listdir(fixation_fpath)

    # key: dicom_id: {correct_data, incorrect_data_class_label...}
    fixation_transcript_data = {}
    fixation_transcript_metadata = {}  # key: dicom_id: {error_info...}

    """
    50 samples divided into 10 samples 
    1. Missed abnormality due to removed transcription sentence
    2. Missed abnormality due to reduced fixation duration
    3. Missed abnormality due to less experience
    4. Missed abnormality due to removed transcription sentence and reduced fixation duration
    5. No error
    """
    num_samples = len(dicom_ids)
    subgroup_ratios = {
        'no_error_ratio': 0.2,
        'class1_ratio': 0.2,
        'class2_ratio': 0.2,
        'class1_and_2_ratio': 0.2,
        'class3_ratio': 0.2,
    }

    subgroup_samples = []
    subgroup_labels = ['no_error', 'class1',
                       'class2', 'class1_and_2', 'class3']
    subgroup_label_count = {
        'no_error': 0,
        'class1': 0,
        'class2': 0,
        'class1_and_2': 0,
        'class3': 0,
    }

    for ratio in subgroup_ratios.values():
        subgroup_samples.append(int(num_samples * ratio))

    curr_subgroup_indx = 0
    dicom_id_indx = 0

    invalid_timestamps_file = "invalid_timestamps.txt"
    with open(invalid_timestamps_file, 'w') as f:
        pass

    for sg in subgroup_samples:
        if dicom_id_indx >= len(dicom_ids):
            break

        sgIndx = 0
        while sgIndx < sg:
            if dicom_id_indx >= len(dicom_ids):
                break

            current_class_label = subgroup_labels[curr_subgroup_indx]
            current_dicom_id = dicom_ids[dicom_id_indx]
            transcript_path = os.path.join(
                audio_seg_transcript_fpath, current_dicom_id)
            fixation_csv_path = os.path.join(
                fixation_fpath, current_dicom_id)

            print("Dicom Id:", current_dicom_id)
            print("Perceptual Error Class Label:",
                  current_class_label)

            dicom_id_fixation_transcript_data = {
                'correct_data': {},
                'incorrect_data': {},
            }

            dicom_id_perceptual_error_metadata = {
                'class_label_1': 0,
                'class_label_2': 0,
                'class_label_3': 0,
                "class_label_1_description": perceptual_error_class_labels[1],
                "class_label_2_description": perceptual_error_class_labels[2],
                "class_label_3_description": perceptual_error_class_labels[3],
                # [{'class_label_1': 0, 'class_label_2': 0, 'class_label_3': 0, 'phrase': "", 'begin_time': 0, 'end_time': 0}]
                'phrases': [],
                'missed_fixation_points': [],
                'fixation_points_duration_reduced': [],
            }

            transcript = json.load(
                open(os.path.join(transcript_path, "transcript.json")))

            if not contains_valid_timestamps(transcript):
                dicom_id_indx += 1

                with open(invalid_timestamps_file, 'a') as f:
                    f.write("INVALID TIMESTAMPS | Dicom Id: " +
                            current_dicom_id + '\n')

                print("INVALID TIMESTAMPS | Dicom Id: " +
                      current_dicom_id + '\n')
                print(
                    "================================================================================")
                continue

            correct_abnormality_transcript = label_abnormality_transcript_with_timestamps(
                transcript)

            if len(correct_abnormality_transcript) == 0:
                dicom_id_indx += 1

                with open(invalid_timestamps_file, 'a') as f:
                    f.write("NO ABNORMALITY TRANSCRIPTION FOUND | Dicom Id: " +
                            current_dicom_id + '\n')

                print("NO ABNORMALITY TRANSCRIPTION FOUND | Dicom Id: " +
                      current_dicom_id + '\n')
                print(
                    "================================================================================")
                continue

            removed_sentence_indx = random.randint(
                0, len(correct_abnormality_transcript) - 1)

            if current_class_label == 'class1':

                c1_missed_fixations_df, c1_remaining_fixations_output_df = create_class1_perceptual_error(
                    fixation_csv_path, removed_sentence_indx, correct_abnormality_transcript)

                c1_removed_sentence_metadata = {
                    'class_label_1': 1,
                    'class_label_2': 0,
                    'class_label_3': 0,
                    'class_label_description': perceptual_error_class_labels[1],
                    'phrase': correct_abnormality_transcript[removed_sentence_indx]['sentence'],
                    'begin_time': correct_abnormality_transcript[removed_sentence_indx]['begin_time'],
                    'end_time': correct_abnormality_transcript[removed_sentence_indx]['end_time'],
                }

                dicom_id_fixation_transcript_data['incorrect_data'] = convert_df_to_dict(
                    c1_remaining_fixations_output_df)

                incorrect_abnormality_transcript = [tr for i, tr in enumerate(
                    correct_abnormality_transcript) if i != removed_sentence_indx]
                dicom_id_fixation_transcript_data['incorrect_data']['transcript'] = incorrect_abnormality_transcript

                dicom_id_perceptual_error_metadata['class_label_1'] = 1
                dicom_id_perceptual_error_metadata['phrases'].append(
                    c1_removed_sentence_metadata)
                dicom_id_perceptual_error_metadata['missed_fixation_points'].append(convert_df_to_dict(
                    c1_missed_fixations_df))
            elif current_class_label == 'class2':

                c2_reduced_fixations_df, c2_reduced_fixations_output_df = create_class2_perceptual_error(fixation_csv_path, removed_sentence_indx,
                                                                                                         correct_abnormality_transcript, half_eye_gaze_fixation)

                dicom_id_fixation_transcript_data['incorrect_data'] = convert_df_to_dict(
                    c2_reduced_fixations_output_df)

                incorrect_abnormality_transcript = [tr for i, tr in enumerate(
                    correct_abnormality_transcript) if i != removed_sentence_indx]
                dicom_id_fixation_transcript_data['incorrect_data']['transcript'] = incorrect_abnormality_transcript

                c2_removed_sentence_metadata = {
                    'class_label_1': 0,
                    'class_label_2': 1,
                    'class_label_3': 0,
                    'class_label_description': perceptual_error_class_labels[2],
                    'phrase': correct_abnormality_transcript[removed_sentence_indx]['sentence'],
                    'begin_time': correct_abnormality_transcript[removed_sentence_indx]['begin_time'],
                    'end_time': correct_abnormality_transcript[removed_sentence_indx]['end_time'],
                }

                dicom_id_perceptual_error_metadata['class_label_2'] = 1
                dicom_id_perceptual_error_metadata['phrases'].append(
                    c2_removed_sentence_metadata)
                dicom_id_perceptual_error_metadata['fixation_points_duration_reduced'].append(convert_df_to_dict(
                    c2_reduced_fixations_df))
            elif current_class_label == 'class1_and_2':
                if len(correct_abnormality_transcript) < 2:
                    dicom_id_indx += 1
                    print(
                        "LESS THAN 2 ABNORMALITY TRANSCRIPTIONS FOR CLASS 1 AND 2 PERCEPTUAL ERROR")
                    print(
                        "================================================================================")
                    continue

                c1_removed_sentence_indx, c2_removed_sentence_indx, c1_c2_missed_fixations_df, c1_c2_remaining_fixations_df, c1_c2_fixations_reduced_df, c1_c2_final_fixations_output_df = create_both_class1_and_class2_perceptual_error(
                    fixation_csv_path, correct_abnormality_transcript, half_eye_gaze_fixation)

                dicom_id_fixation_transcript_data['incorrect_data'] = convert_df_to_dict(
                    c1_c2_final_fixations_output_df)

                incorrect_abnormality_transcript = [tr for i, tr in enumerate(
                    correct_abnormality_transcript) if i != c1_removed_sentence_indx and i != c2_removed_sentence_indx]
                dicom_id_fixation_transcript_data['incorrect_data']['transcript'] = incorrect_abnormality_transcript

                c1_removed_sentence_metadata = {
                    'class_label_1': 1,
                    'class_label_2': 0,
                    'class_label_3': 0,
                    'class_label_description': perceptual_error_class_labels[1],
                    'phrase': correct_abnormality_transcript[c1_removed_sentence_indx]['sentence'],
                    'begin_time': correct_abnormality_transcript[c1_removed_sentence_indx]['begin_time'],
                    'end_time': correct_abnormality_transcript[c1_removed_sentence_indx]['end_time'],
                }
                c2_removed_sentence_metadata = {
                    'class_label_1': 0,
                    'class_label_2': 1,
                    'class_label_3': 0,
                    'class_label_description': perceptual_error_class_labels[2],
                    'phrase': correct_abnormality_transcript[c2_removed_sentence_indx]['sentence'],
                    'begin_time': correct_abnormality_transcript[c2_removed_sentence_indx]['begin_time'],
                    'end_time': correct_abnormality_transcript[c2_removed_sentence_indx]['end_time'],
                }

                dicom_id_perceptual_error_metadata['class_label_1'] = 1
                dicom_id_perceptual_error_metadata['class_label_2'] = 1
                dicom_id_perceptual_error_metadata['missed_fixation_points'].append(convert_df_to_dict(
                    c1_c2_missed_fixations_df))
                dicom_id_perceptual_error_metadata['fixation_points_duration_reduced'].append(convert_df_to_dict(
                    c1_c2_fixations_reduced_df))
                dicom_id_perceptual_error_metadata['phrases'].append(
                    c1_removed_sentence_metadata)
                dicom_id_perceptual_error_metadata['phrases'].append(
                    c2_removed_sentence_metadata)
            elif current_class_label == 'class3':
                c3_fixations_df = create_class3_perceptual_error(
                    fixation_csv_path)

                dicom_id_fixation_transcript_data['incorrect_data'] = convert_df_to_dict(
                    c3_fixations_df)

                incorrect_abnormality_transcript = [tr for i, tr in enumerate(
                    correct_abnormality_transcript) if i != removed_sentence_indx]
                dicom_id_fixation_transcript_data['incorrect_data']['transcript'] = incorrect_abnormality_transcript

                dicom_id_perceptual_error_metadata['class_label_3'] = 1
                c3_removed_sentence_metadata = {
                    'class_label_1': 0,
                    'class_label_2': 0,
                    'class_label_3': 1,
                    'class_label_description': perceptual_error_class_labels[3],
                    'phrase': correct_abnormality_transcript[removed_sentence_indx]['sentence'],
                    'begin_time': correct_abnormality_transcript[removed_sentence_indx]['begin_time'],
                    'end_time': correct_abnormality_transcript[removed_sentence_indx]['end_time'],
                }
                dicom_id_perceptual_error_metadata['phrases'].append(
                    c3_removed_sentence_metadata)

            fixation_transcript_data[current_dicom_id] = dicom_id_fixation_transcript_data
            fixation_transcript_metadata[current_dicom_id
                                         ] = dicom_id_perceptual_error_metadata

            correct_data_df = get_correct_data(fixation_csv_path)
            dicom_id_fixation_transcript_data['correct_data'] = convert_df_to_dict(
                correct_data_df)
            dicom_id_fixation_transcript_data['correct_data']['transcript'] = correct_abnormality_transcript

            dicom_id_indx += 1
            sgIndx += 1
            subgroup_label_count[current_class_label] += 1
            print(
                "================================================================================")

        curr_subgroup_indx += 1

    with open('original_fixation_transcript_data.json', 'w') as json_file:
        json_file.write(json.dumps(fixation_transcript_data, indent=4))

    with open('original_fixation_transcript_metadata.json', 'w') as json_file:
        json_file.write(json.dumps(
            fixation_transcript_metadata, indent=4))

    end_time = time.time()
    print(
        f'Output Data Size: Fixation Transcript = {len(fixation_transcript_data)} | Metadata = {len(fixation_transcript_metadata)}')
    print(subgroup_label_count)
    print(f'Elapsed Time: {end_time - start_time} seconds')


if __name__ == '__main__':
    main()
