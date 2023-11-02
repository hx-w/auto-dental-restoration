# -*- coding: utf-8 -*-

from method import Mtd_DIFNet, Mtd_ToothDIT
from ui import GradioGUI


if __name__ == '__main__':
    gradio_inst = GradioGUI(
        [
            Mtd_DIFNet('models/DIF/config.yml'),
            Mtd_ToothDIT('models/ToothDIT/config.yml')
        ]
    )

    gradio_inst.launch_public()
