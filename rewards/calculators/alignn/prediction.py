# This code is adapted from https://github.com/usnistgov/alignn/blob/main/alignn/pretrained.py
import os
import json
import zipfile
import tempfile
import requests
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from jarvis.core.atoms import Atoms
from alignn.graphs import Graph
from alignn.dataset import get_torch_dataset
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from huggingface_hub import hf_hub_download


"""
Name of the model, figshare link, number of outputs,
extra config params (optional)
"""
# See also, alignn/ff/ff.py
# Both alignn and alignn_atomwise
# models are shared

# See: alignn/ff/all_models_alignn.json
# to load as a calculator
ALIGNN_MODEL_LIST = {
    "jv_formation_energy_peratom_alignn": [
        "https://figshare.com/ndownloader/files/31458679",
        1,
    ],
    "jv_optb88vdw_total_energy_alignn": [
        "https://figshare.com/ndownloader/files/31459642",
        1,
    ],
    "jv_optb88vdw_bandgap_alignn": [
        "https://figshare.com/ndownloader/files/31459636",
        1,
    ],
    "jv_mbj_bandgap_alignn": [
        "https://figshare.com/ndownloader/files/31458694",
        1,
    ],
    "jv_spillage_alignn": [
        "https://figshare.com/ndownloader/files/31458736",
        1,
    ],
    "jv_slme_alignn": ["https://figshare.com/ndownloader/files/31458727", 1],
    "jv_bulk_modulus_kv_alignn": [
        "https://figshare.com/ndownloader/files/31458649",
        1,
    ],
    "jv_shear_modulus_gv_alignn": [
        "https://figshare.com/ndownloader/files/31458724",
        1,
    ],
    "jv_n-Seebeck_alignn": [
        "https://figshare.com/ndownloader/files/31458718",
        1,
    ],
    "jv_n-powerfact_alignn": [
        "https://figshare.com/ndownloader/files/31458712",
        1,
    ],
    "intermat_cbm": [
        "https://figshare.com/ndownloader/files/45392908",
        1,
    ],
    "intermat_vbm": [
        "https://figshare.com/ndownloader/files/45392914",
        1,
    ],
    "intermat_phi": [
        "https://figshare.com/ndownloader/files/45392911",
        1,
    ],
    "jv_magmom_oszicar_alignn": [
        "https://figshare.com/ndownloader/files/31458685",
        1,
    ],
    "jv_kpoint_length_unit_alignn": [
        "https://figshare.com/ndownloader/files/31458682",
        1,
    ],
    "jv_avg_elec_mass_alignn": [
        "https://figshare.com/ndownloader/files/31458643",
        1,
    ],
    "jv_avg_hole_mass_alignn": [
        "https://figshare.com/ndownloader/files/31458646",
        1,
    ],
    "jv_epsx_alignn": ["https://figshare.com/ndownloader/files/31458667", 1],
    "jv_mepsx_alignn": ["https://figshare.com/ndownloader/files/31458703", 1],
    "jv_max_efg_alignn": [
        "https://figshare.com/ndownloader/files/31458691",
        1,
    ],
    "jv_ehull_alignn": ["https://figshare.com/ndownloader/files/31458658", 1],
    "jv_dfpt_piezo_max_dielectric_alignn": [
        "https://figshare.com/ndownloader/files/31458652",
        1,
    ],
    "jv_dfpt_piezo_max_dij_alignn": [
        "https://figshare.com/ndownloader/files/31458655",
        1,
    ],
    "jv_exfoliation_energy_alignn": [
        "https://figshare.com/ndownloader/files/31458676",
        1,
    ],
    "jv_supercon_tc_alignn": [
        "https://figshare.com/ndownloader/files/38789199",
        1,
    ],
    "jv_supercon_edos_alignn": [
        "https://figshare.com/ndownloader/files/39946300",
        1,
    ],
    "jv_supercon_debye_alignn": [
        "https://figshare.com/ndownloader/files/39946297",
        1,
    ],
    "jv_supercon_a2F_alignn": [
        "https://figshare.com/ndownloader/files/38801886",
        100,
    ],
    "mp_e_form_alignn": [
        "https://figshare.com/ndownloader/files/31458811",
        1,
    ],
    "mp_gappbe_alignn": [
        "https://figshare.com/ndownloader/files/31458814",
        1,
    ],
    "tinnet_O_alignn": ["https://figshare.com/ndownloader/files/41962800", 1],
    "tinnet_N_alignn": ["https://figshare.com/ndownloader/files/41962797", 1],
    "tinnet_OH_alignn": ["https://figshare.com/ndownloader/files/41962803", 1],
    "AGRA_O_alignn": ["https://figshare.com/ndownloader/files/41966619", 1],
    "AGRA_OH_alignn": ["https://figshare.com/ndownloader/files/41966610", 1],
    "AGRA_CHO_alignn": ["https://figshare.com/ndownloader/files/41966643", 1],
    "AGRA_CO_alignn": ["https://figshare.com/ndownloader/files/41966634", 1],
    "AGRA_COOH_alignn": ["https://figshare.com/ndownloader/41966646", 1],
    "qm9_U0_alignn": ["https://figshare.com/ndownloader/files/31459054", 1],
    "qm9_U_alignn": ["https://figshare.com/ndownloader/files/31459051", 1],
    "qm9_alpha_alignn": ["https://figshare.com/ndownloader/files/31459027", 1],
    "qm9_gap_alignn": ["https://figshare.com/ndownloader/files/31459036", 1],
    "qm9_G_alignn": ["https://figshare.com/ndownloader/files/31459033", 1],
    "qm9_HOMO_alignn": ["https://figshare.com/ndownloader/files/31459042", 1],
    "qm9_LUMO_alignn": ["https://figshare.com/ndownloader/files/31459045", 1],
    "qm9_ZPVE_alignn": ["https://figshare.com/ndownloader/files/31459057", 1],
    "hmof_co2_absp_alignn": [
        "https://figshare.com/ndownloader/files/31459198",
        5,
    ],
    "hmof_max_co2_adsp_alignn": [
        "https://figshare.com/ndownloader/files/31459207",
        1,
    ],
    "hmof_surface_area_m2g_alignn": [
        "https://figshare.com/ndownloader/files/31459222",
        1,
    ],
    "hmof_surface_area_m2cm3_alignn": [
        "https://figshare.com/ndownloader/files/31459219",
        1,
    ],
    "hmof_pld_alignn": ["https://figshare.com/ndownloader/files/31459216", 1],
    "hmof_lcd_alignn": ["https://figshare.com/ndownloader/files/31459201", 1],
    "hmof_void_fraction_alignn": [
        "https://figshare.com/ndownloader/files/31459228",
        1,
    ],
    "ocp2020_all": ["https://figshare.com/ndownloader/files/41411025", 1],
    "ocp2020_100k": ["https://figshare.com/ndownloader/files/41967303", 1],
    "ocp2020_10k": ["https://figshare.com/ndownloader/files/41967330", 1],
    "jv_pdos_alignn": [
        "https://figshare.com/ndownloader/files/36757005",
        66,
        {"alignn_layers": 6, "gcn_layers": 6},
    ],
}


HF_MODEL_DICT = {
    'mp_bandgap_hf': 'prop_pred/alignn/band_gap',
    'mp_bulk_modulus_hf': 'prop_pred/alignn/bulk_modulus_voigt',
    'mp_piezoelectric_hf': 'prop_pred/alignn/piezoelectric_modulus',
    'mp_shear_modulus_hf': 'prop_pred/alignn/shear_modulus_voigt',
    'mp_dielectric_hf': 'prop_pred/alignn/total_dielectric_constant',
    'mp_total_mag_hf': 'prop_pred/alignn/total_magnetization',
    'mp_total_mag_per_atom_hf': 'prop_pred/alignn/total_magnetization_per_atom',
    'mp_vickers_hardness_hf': 'prop_pred/alignn/vickers_hardness',
}


def get_device(device: str | None = None):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    return torch.device(device)


def get_figshare_model(
    model_name="jv_formation_energy_peratom_alignn",
    device=None,
):
    device = get_device(device)

    """Get ALIGNN torch models from figshare."""
    # https://figshare.com/projects/ALIGNN_models/126478

    tmp = ALIGNN_MODEL_LIST[model_name]
    url = tmp[0]
    # output_features = tmp[1]
    # if len(tmp) > 2:
    #    config_params = tmp[2]
    # else:
    #    config_params = {}
    zfile = model_name + ".zip"
    path = str(os.path.join(os.path.dirname(__file__), zfile))
    if not os.path.isfile(path):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes, unit="iB", unit_scale=True
        )
        with open(path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    zp = zipfile.ZipFile(path)
    names = zp.namelist()
    chks = []
    cfg = []
    for i in names:
        if "checkpoint_" in i and "pt" in i:
            tmp = i
            chks.append(i)
        if "config.json" in i:
            cfg = i
        if "best_model.pt" in i:
            tmp = i
            chks.append(i)

    # print("Using chk file", tmp, "from ", chks)
    # print("Path", os.path.abspath(path))
    # print("Config", os.path.abspath(cfg))
    config = json.loads(zipfile.ZipFile(path).read(cfg))
    # print("Loading the zipfile...", zipfile.ZipFile(path).namelist())
    data = zipfile.ZipFile(path).read(tmp)
    # model = ALIGNN(
    #    ALIGNNConfig(
    #        name="alignn", output_features=output_features, **config_params
    #    )
    # )
    model = ALIGNN(ALIGNNConfig(**config["model"]))

    new_file, filename = tempfile.mkstemp()
    with open(filename, "wb") as f:
        f.write(data)
    model.load_state_dict(torch.load(filename, map_location=device)["model"])
    model.to(device)
    model.eval()
    if os.path.exists(filename):
        os.remove(filename)
    return model


def get_huggingface_model(
    model_name="mp_bandgap_hf",
    device=None,
):
    device = get_device(device)

    """Get ALIGNN torch models from huggingface."""

    hf_folder = HF_MODEL_DICT[model_name]
    ckpt_path = hf_hub_download(
            repo_id="jwchen25/MatInvent",
            filename=f"{hf_folder}/best_model.pt",
        )
    cfg_path = hf_hub_download(
        repo_id="jwchen25/MatInvent",
        filename=f"{hf_folder}/config.json",
    )

    with open(cfg_path, 'r') as file:
        config = json.load(file)
    model = ALIGNNAtomWise(ALIGNNAtomWiseConfig(**config["model"]))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_model(
    model_name="mp_bandgap_hf",
    device=None,
):
    if '_hf' in model_name:
        model = get_huggingface_model(model_name, device)
    else:
        model = get_figshare_model(model_name, device)
    return model


def get_prediction(
    model_name="jv_formation_energy_peratom_alignn",
    atoms=None,
    cutoff=8,
    max_neighbors=12,
    device=None,
):
    device = get_device(device)

    """Get model prediction on a single structure."""
    model = get_model(model_name)
    # print("Loading completed.")
    g, lg = Graph.atom_dgl_multigraph(
        atoms,
        cutoff=float(cutoff),
        max_neighbors=max_neighbors,
    )
    lat = torch.tensor(atoms.lattice_mat)
    out_data = (
        model([g.to(device), lg.to(device), lat.to(device)])
        .detach()
        .cpu()
        .numpy()
        .flatten()
        .tolist()
    )
    return out_data


def get_multiple_predictions(
    atoms_array=[],
    jids=[],
    cutoff=8,
    neighbor_strategy="k-nearest",
    max_neighbors=12,
    use_canonize=True,
    target="prop",
    atom_features="cgcnn",
    line_graph=True,
    workers=0,
    filename="pred_data.json",
    include_atoms=True,
    pin_memory=False,
    output_features=1,
    batch_size=64,
    model=None,
    model_name="jv_formation_energy_peratom_alignn",
    print_freq=100,
    device=None,
):
    device = get_device(device)
    if '_hf' in model_name:
        batch_size = 1

    """Use pretrained model on a number of structures."""
    # if use_lmdb:
    #    print("Using LMDB dataset.")
    #    from alignn.lmdb_dataset import get_torch_dataset
    # else:
    #    print("Not using LMDB dataset, memory footprint maybe high.")
    #    from alignn.dataset import get_torch_dataset

    # import glob
    # atoms_array=[]
    # for i in glob.glob("alignn/examples/sample_data/*.vasp"):
    #      atoms=Atoms.from_poscar(i)
    #      atoms_array.append(atoms)
    # get_multiple_predictions(atoms_array=atoms_array)
    if not jids:
        jids = ["id-" + str(i) for i in np.arange(len(atoms_array))]
    mem = []
    for i, ii in tqdm(enumerate(atoms_array), total=len(atoms_array)):
        info = {}
        if isinstance(ii, Atoms):
            ii = ii.to_dict()
        info["atoms"] = ii  # .to_dict()
        info["prop"] = -9999  # place-holder only
        info["jid"] = jids[i]  # str(i)
        mem.append(info)

    if model is None:
        try:
            model = get_model(model_name)
        except Exception as exp:
            raise ValueError(
                'Check is the model name exists using "pretrained.py -h"', exp
            )
            pass

    # Note cut-off is usually 8 for solids and 5 for molecules
    def atoms_to_graph(atoms):
        """Convert structure dict to DGLGraph."""
        structure = Atoms.from_dict(atoms)
        return Graph.atom_dgl_multigraph(
            structure,
            cutoff=cutoff,
            atom_features="atomic_number",
            max_neighbors=max_neighbors,
            compute_line_graph=True,
            use_canonize=use_canonize,
        )

    test_data = get_torch_dataset(
        dataset=mem,
        target="prop",
        neighbor_strategy=neighbor_strategy,
        atom_features=atom_features,
        use_canonize=use_canonize,
        line_graph=line_graph,
    )

    collate_fn = test_data.collate_line_graph
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    results = []
    with torch.no_grad():
        for dat in test_loader:
            g, lg, lat, target = dat
            out_data = model([g.to(device), lg.to(device), lat.to(device)])
            if '_hf' in model_name:
                out_data = out_data['out']
            out_data = out_data.cpu().numpy().tolist()
            if batch_size == 1:
                out_data = [out_data]
            results.extend(out_data)
    return results
