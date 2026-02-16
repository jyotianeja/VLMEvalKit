#open excel file and print first line

import pandas as pd
#file = '/home/mharrison/repos/VLMEvalKit/outputs/phi4-personas-8017/phi4-personas-8017_ScreenSpot_v2_Web.xlsx'
file = '/home/mharrison/repos/VLMEvalKit/outputs/phi4-seeclickweb-8017/phi4-seeclickweb-8017_ScreenSpot_v2_Web.xlsx'
#file = '/home/mharrison/repos/VLMEvalKit/outputs/phi4mm-8011/phi4mm-8011_ScreenSpot_v2_Web.xlsx'
df = pd.read_excel(file)
print(df.iloc[7])
