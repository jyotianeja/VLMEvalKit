from vlmeval.dataset import IMAGE_DATASET
import os

# file_names = [f for f in os.listdir('/home/mharrison/repos/VLMEvalKit/vlmeval/dataset') if f.endswith('.py')]


# base_dir = '/home/mharrison/repos/VLMEvalKit/vlmeval/dataset'
# output_file = 'image_dataset_names.csv'
# for file_name in file_names:
#     file_path = f'{base_dir}/{file_name}'
#     # import as python
#     module_name = file_name[:-3]
#     module = __import__(f'vlmeval.dataset.{module_name}', fromlist=[module_name])

#     class_list = [getattr(module, attr) for attr in dir(module) if isinstance(getattr(module, attr), type)]
#     for cls in class_list:
#         class_name = cls.__name__
#         if class_name not in IMAGE_DATASET:
#             continue
#         if hasattr(cls, 'DATASET_URL'):
#             dataset_urls = getattr(cls, 'DATASET_URL')
#             if isinstance(dataset_urls, dict) and len(dataset_urls) > 0:
#                 dataset_names = list(dataset_urls.keys())
#                 for dataset_name in dataset_names:
#                     with open(output_file, 'a') as f:
#                         f.write(f'{module_name},{cls.__name__},{dataset_name}\n')



