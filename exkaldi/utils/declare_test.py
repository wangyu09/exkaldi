
# coding=utf-8
#
# Yu Wang (University of Yamanashi)
# May, 2020
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Tests for exkaldi.utils.utils.declare'''

from exkaldi.utils import declare

def test_is_classes_and_belong_classes():

  class A:
    def __init__(self):
      pass
  
  class B(A):
    def __init__(self):
      pass
  
  b = B()

  declare.is_classes("test object",b,B)
  declare.belong_classes("test object",b,A)
