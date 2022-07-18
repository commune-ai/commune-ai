
from copy import deepcopy
import sys
import os
import datetime
sys.path.append(os.getenv('PWD'))

from commune.process import BaseProcess


class DemoModule(BaseProcess):
    default_cfg_path=f"{os.environ['PWD']}/commune/config/contract/commune/base/demo.yaml"
    def process(self, **kwargs):
        for i in range(10):
            self.module['commune'].run(override={'cfg.name': f'Model-a-{i}', 'cfg.refresh': False})

if __name__ == "__main__":
    module = DemoModule.deploy(actor=False)
    module.run()
    

       # print(portfolio.valueRatios)


    

    