from edges import types as tp
from .cal_loads import get_loads11_from_load_and_switch
from .s11model import S11Model
from edges.io import LoadS11, Calkit, SwitchingState

def get_s11model_from_edges2_ants11_files(
    files: tuple[tp.PathLike, tp.PathLike, tp.PathLike, tp.PathLike],
    switchdef: SwitchingState,
    **kwargs
    
) -> S11Model:
    return get_loads11_from_load_and_switch(
        loaddef=LoadS11(
            calkit = Calkit(
                open=files[0],
                short=files[1],
                match=files[2]
            ), 
            external=files[3]
        ),
        switchdef = switchdef,
        **kwargs
    )
