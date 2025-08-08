from pathlib import Path
from local_information import OpenSystem

checkpoint_folder = (
    "./data/NV_diffusion=0.0_J=-0.1_L=501_rtrunc=0.2_mean_dis=5.0_delta_phi=0.1"
)

working_dir = Path(__file__).parent.resolve().as_posix()
system = OpenSystem.from_checkpoint(
    folder=checkpoint_folder,
    module_path=working_dir + "/nv_gradient_field_diffusion.py",
)
