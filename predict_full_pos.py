# This function is for making a prediction from a single image, or a folder of images
# predict_file prints result summary to console and returns a FilePredictionResult object
# predict_dir runs predictfile, and returns a FolderPredictionResult. It also saves a json file in the image folder with the same info.

# Most fields in the return values should self-explanatory, except:
# list_of_errors_by_piece - this returns a string of two chars, the first being the true piece, the second what that piece was mistaken for. It also returns a count for each such error.

# See the bottom of the file for an example of use/test


import datetime, os, json, numpy as np, re
from typing import List
from dataclasses import dataclass, field
from collections import defaultdict
import keras
import zoom_to_position as zoom
import matplotlib.image as mpimg
from FEN_to_64grid import FEN_to_seq, seq_to_FEN
import char_enum_pieces
from typing import Dict

@dataclass
class FilePredictionResult:
    errors_num: int = 0
    err_positions: List[int] = field(default_factory=list)      # Note: 0-indexed, runs from 0-63 (to be used as indexes into string)
    err_pieces: List[str] = field(default_factory=list)

@dataclass
class FolderPredictionResult:

    # Metadata
    timestamp: datetime.datetime
    model: str
    folder: str
    # By file
    total_files: int = 0
    total_files_w_errors: int = 0
    error_ratio_by_files: float = .0
    file_ratio_of_total_by_error_ratio: Dict[float, float] = field(
        default_factory=lambda: {key: 0 for key in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    )
    # By fields
    total_fields: int = 0
    total_fields_w_errors: int = 0
    error_ratio_by_fields: float = .0
    # By piece
    list_of_pieces_in_truth: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    list_of_errors_by_piece: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_ratio_by_pieces: defaultdict[str, float] = field(default_factory=lambda: defaultdict(int))
    # File list
    file_list_by_error_ratio: Dict[float, List[str]] = field(
        default_factory=lambda: {key: [] for key in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    )

class predictor:
    def __init__(self, modelpath):
        self.model = keras.models.load_model(modelpath)
        self.modelpath = modelpath
    def predict_file(self, filename):
        # Init
        rv = FilePredictionResult(0, [], [])
        filename_wo_ending = filename.replace('.jpeg', '')                  # Removing last name
        FEN_string_true = re.sub(r".*[/\\]", "", filename_wo_ending)        # Removing everything before last folder delimiter
        seq_string_true = FEN_to_seq(FEN_string_true)
        field_image_array = []
        image = mpimg.imread(filename)

        # Construct matrix to predict on

        for position in range(1,65) :
            field_image = np.array(zoom.zoom_to_position(np.array(image), position))
            field_image = field_image.reshape(50, 50, 3).astype('float32') / 255
            field_image_array.append( field_image )
        field_image_array = np.array(field_image_array)

        # Make prediction
        prediction_array_raw = self.model.predict(field_image_array)

        # Construct sequence array from prediction array
        seq_string_prediction_nums = np.argmax(prediction_array_raw, axis=1)
        seq_string_prediction = ""
        for position in range(0,64):
            seq_string_prediction += char_enum_pieces.CharEnum(seq_string_prediction_nums[position]).name

        # Note errors
        for i in range(0,64):
            if (seq_string_true[i] != seq_string_prediction[i]):
                rv.errors_num += 1
                rv.err_positions.append(i)
                rv.err_pieces.append( str(seq_string_true[i]) + str(seq_string_prediction[i]) )

        return rv

    def predict_dir(self, directorypath):
        rv = FolderPredictionResult(timestamp = datetime.datetime.now(), folder=directorypath, model=self.modelpath)
        last_file_result = FilePredictionResult
        files_by_error_ratio_count = {key: 0 for key in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

        for file in os.listdir(directorypath):
            if file.endswith(".jpeg"):
                # Get file prediction
                file_w_path = os.path.join(directorypath, file)
                rv.total_files += 1
                rv.total_fields += 64
                last_file_result = self.predict_file(file_w_path)

                # Populate simple return values
                for char in FEN_to_seq(file.replace('.jpeg', '')):
                    rv.list_of_pieces_in_truth[char] += 1
                if (last_file_result.errors_num != 0):
                    rv.total_files_w_errors += 1
                    rv.total_fields_w_errors += last_file_result.errors_num
                    for error_piece in last_file_result.err_pieces:
                        rv.list_of_errors_by_piece[error_piece] += 1                        
                ratio_error = np.round(last_file_result.errors_num / 64, decimals=1)
                files_by_error_ratio_count[ratio_error] += 1
                rv.file_list_by_error_ratio[ratio_error].append(file)


        # Generate return value ratios, etc.
        for ratio, count in files_by_error_ratio_count.items():
            rv.file_ratio_of_total_by_error_ratio[ratio] = np.round(count / rv.total_files, decimals=2)
        rv.error_ratio_by_files = rv.total_files_w_errors / rv.total_files
        rv.error_ratio_by_fields = rv.total_fields_w_errors / rv.total_fields

        total_errors_for_piece = defaultdict(int)
        for key, value in rv.list_of_errors_by_piece.items():
            true_char_that_lead_to_error = key[0]
            total_errors_for_piece[true_char_that_lead_to_error] += value
        for key, value in total_errors_for_piece.items():
            rv.error_ratio_by_pieces[key] += value / rv.list_of_pieces_in_truth[key]
            

        # Save as json

        savefile_fullpath = os.path.join(directorypath, str(rv.timestamp.isoformat()) + ".json")
        savefile_fullpath = re.sub(r":", "-", savefile_fullpath)        # : to - to help Windows

        with open(savefile_fullpath, "w" ) as savefile:
            json.dump(vars(rv), savefile, default=json_serializer, indent=4)

        print(f"Saved output to: {savefile_fullpath}" )        
        return rv


def json_serializer(object):
    if isinstance(object, datetime.datetime):
        return object.isoformat()
    if isinstance(object, defaultdict):
        return dict(object)



# For testing:

#predictor = predictor("Done model and execution code/advnced_model.keras")      # Model .keras file relative path
#full_output = predictor.predict_dir("chessboard_example_images")                # Folder to search for .jpeg files
# print(full_output)                                                            # Output not needed, above saves to json
