import os 
from pathlib import Path



class Pathfinder:

    def __init__(
            self,
            gwak_env: str,  
            suffix: str = None, 
            file_name: str = None, 
        ):

        self.file_name = file_name

        if suffix is not None:
            self.path = Path(os.getenv(gwak_env)) / f"{suffix}"

        else: 
            self.path = Path(os.getenv(gwak_env))


    def get_path(self):
            
        self.path.mkdir(parents=True, exist_ok=True)

        if self.file_name is not None:

            return self.path / self.file_name
        
        return self.path




# class GWAK_CONTAINER_ROOT(Pathfinder):

#     def __init__(self):
#         self.path = os.getenv("GWAK_CONTAINER_ROOT")

# class GWAK_OUTPUT_DIR(Pathfinder):

#     def __init__(self):
#         self.path = os.getenv("GWAK_OUTPUT_DIR")

# class GWAK_TRITON_DIR(Pathfinder):

#     def __init__(self):
#         self.path = os.getenv("GWAK_TRITON_DIR")



# class GWAK_SOME_DATA_DIR(Pathfinder):

#     def __init__(self, file_name):
#         self.path = Path("/home/hongyin.chen/Data/gwak/") / file_name