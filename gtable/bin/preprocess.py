# # -*- coding: utf-8 -*-
# # Copyright 2020 Unknot.id Inc. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
#
# from gtable.data.inputter import pickle_save
# from gtable.data.dataset import CSVDataset
# from gtable.data.dataset import get_data_loader
#
#
# class Preprocess:
#     def __init__(self, ctx):
#         self.context = ctx
#         self.logging = ctx.logger
#         self.config = ctx.config
#
#     def save_dataset(self, data, corpus_type):
#         pickle_save(self.context, data, self.config.save_data + f'.{corpus_type}' + '.pkl')
#
#     def run(self, inputPath, isSave=True):
#         data_loader = get_data_loader(self.context)
#
#         data = data_loader(inputPath)
#
#         if isSave:
#             self.save_dataset(data_loader, self.context.data_type)
#
#         return data_loader
#
#
