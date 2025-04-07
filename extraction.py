import pathlib
from pprint import pprint
from timspy.df import TimsPyDF
import pandas as pd

all_columns = ('frame', 'intensity', 'mz', 'inv_ion_mobility')

def process_data(file_path):
    """
        Processes IM-MSI data

        Parameters:
            file_path (str or pathlib.Path): Path to the .d file containing IM-MSI data.
        Returns:
            tuple: (TimsPyDF object, max_frame)
    """
    path = pathlib.Path(file_path)
    print(f'Processing data from {path}')
    D = TimsPyDF(path)
    queried_data = D.query(frames=D.ms1_frames, columns=all_columns).to_numpy()
    column_frame = D.query(frames=D.ms1_frames, columns=['frame'])
    max_frame = column_frame['frame'].max()
    return queried_data, max_frame  # Return dataset object and max frame number

if __name__ == "__main__":
    file_path = "/path/to/your/data.d"   # Change this to your actual .d file path
    data_array, max_frame = process_data(file_path)

    output_dir = pathlib.Path("RawData")
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, max_frame + 1):
        frame_data = data_array[data_array[:, 0] == i]
        if frame_data.size > 0:
            df = pd.DataFrame(frame_data, columns=all_columns)
            df.to_csv(output_dir / f"{i}.csv", index=False, header=False)
