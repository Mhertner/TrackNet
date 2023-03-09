import pandas as pd
from pathlib import Path
import os


class PreProcessCsv:

    def __init__(self):
        self.checkpoint = None
        #pass

    def run(self, input_path):

        """
        Takes a csv as input and calculates values for null values
        based on known values of the row before and after the null values
        """

        data = pd.read_csv(input_path)

        results = [[0]] * len(data)
        started = False
        old_curr = (0, 0)
        has_old_current = False
        nulls = []

        for idx, row in data.iterrows():
            x = row[1]
            y = row[2]

            # Starting at first point where ball is found
            if not started:
                if row[1:].isna().any():

                    out = [row[0],0,0]
                    results[int(row[0])] = out
                    # results.append(list(row))
                    continue
                else:
                    started = True

            # check if values is not null
            if not row[1:].isna().any():

                curr_cor = (x, y)
                self.checkpoint = int(row[0])
                results[int(row[0])] = list(row)

                # print('FIRST CURR', curr_cor)
                # print('FIRST OLD', old_curr)
                # If we don't have an old value yet, we create it here
                if old_curr == (0, 0):
                    # print('GET TO OLD')
                    # nulls = []
                    old_curr = curr_cor

                # Checks if we have any null values and calculates the new values for these
                if nulls:
                    length = len(nulls) + 1
                    # print('OLD : ', old_curr[0])
                    # print('NEW : ', curr_cor[0])
                    x_discrepancy = max(old_curr[0], curr_cor[0]) - min(old_curr[0], curr_cor[0])
                    y_discrepancy = max(old_curr[1], curr_cor[1]) - min(old_curr[1], curr_cor[1])
                    # print('X', x_discrepancy)
                    # print('Y', y_discrepancy)

                    x_step = x_discrepancy / length
                    y_step = y_discrepancy / length

                    for index, val in enumerate(nulls):
                        # print('INDICES : ', idx, index)
                        # print(val)
                        multiplier = index + 1
                        if old_curr[0] < curr_cor[0]:
                            x = int(old_curr[0] + x_step * multiplier)
                            y = int(old_curr[1] + x_step * multiplier)
                        else:
                            x = int(old_curr[0] - x_step * multiplier)
                            y = int(old_curr[1] - x_step * multiplier)
                        # print('CALCULATED X', x)
                        # print('CALCULATED Y', y)

                        res = [val[0], x, y]
                        # print('REEES', res)
                        results[int(val[0])] = res
                        # print('FINAL : ', results[int(val[0])])
                    nulls = []

                # Appends null-rows to a list
            else:
                # print('ELSE')
                # old_curr = (0,0)
                nulls.append(list(row))
                # print(nulls)

        return pd.DataFrame(results, columns=('Frame', 'X-coordinate', 'Y-coordinate'))


    def save_csv(self, input_path, file_name='', output_path='predictions/'):

        """
        Runs the run methods and saves the dataframe as a csv
        at the specified folder. If the folder is not found it is created
        """

        if file_name == '':
            file_name = input_path

        path = str(output_path + file_name)
        df = self.run(Path(input_path))
        # Check for rows left at the bottom
        print(self.checkpoint)
        for i in range(self.checkpoint, len(df)):
            idx = i
            df['Frame'].loc[[i]] = i
        df2 = df.fillna(0)


        if not Path(output_path).exists():
            os.mkdir(output_path)

        df2.to_csv(Path(path), index=False)