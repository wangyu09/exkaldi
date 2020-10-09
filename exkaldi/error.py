# coding=utf-8
#
# Yu Wang (University of Yamanashi)
# Oct, 2020
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

'''ExKaldi Exception Classes'''

class BaseError(Exception):

  def __init__(self,main,detail=None):
    if detail:
      main += ("\n"+detail)
    super().__init__(main)

class WrongPath(BaseError):pass
class WrongOperation(BaseError):pass
class WrongDataFormat(BaseError):pass
class ShellProcessError(BaseError):pass
class KaldiProcessError(BaseError):pass
class KenlmProcessError(BaseError):pass
class UnsupportedType(BaseError):pass
class UnsupportedKaldiVersion(BaseError): pass