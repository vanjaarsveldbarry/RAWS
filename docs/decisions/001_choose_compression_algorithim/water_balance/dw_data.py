import json
import ssl
import sys
from pathlib import Path

import xarray as xr
from irods.models import Collection, DataObject
from irods.session import iRODSSession
from tqdm import tqdm


def setup_iRodsSession(env_config, password):
    
    def get_irods_environment(irods_environment_file="irods_environment.json"):
        """Reads the irods_environment.json file, which contains the environment
        configuration."""
        with open(irods_environment_file, 'r') as f:
            return json.load(f)

    def setup_session(irods_environment_config,  password, require_ssl = True, ca_file = "/home/barrygwt/.irods/cacert.pem"):
        """Use irods environment files to configure a iRODSSession"""

        if require_ssl:
            ssl_context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH, cafile=ca_file, capath=None, cadata=None)
            ssl_settings = {'client_server_negotiation': 'request_server_negotiation',
                            'client_server_policy': 'CS_NEG_REQUIRE',
                            'encryption_algorithm': 'AES-256-CBC',
                            'encryption_key_size': 32,
                            'encryption_num_hash_rounds': 16,
                            'encryption_salt_size': 8,
                            'ssl_context': ssl_context}
            session = iRODSSession(
                irods_password=password,
                **irods_environment_config,
                **ssl_settings
            )
        else:
            session = iRODSSession(
                password=password,
                **irods_environment_config,
            )

        return session

    session = setup_session(get_irods_environment(env_config), password)
    if session is None:
        print("Error: unable to create session.")
        sys.exit(1)
    else:
        return session

# # Create a session
data_dir = Path(sys.argv[1])
env_config = '/home/barrygwt/.irods/irods_environment_DAG.json'
password = "MbRuH9Utf7h9myOK4xklBxliJJEi9M59"
projectDirectory = "/nluu14p/home/vault-pilot/pcrglobwb_global_30arcsec[1746459966]/original/output"

variables = [
        "totalEvaporation_monthTot_output.nc",
            "precipitation_monthTot_output.nc",
                "totalRunoff_monthTot_output.nc",
                ]

def download_from_yoda(env_config, password, data_dir, projectDirectory, variables):
    savePath_yoda = (data_dir / 'perClone').as_posix()

    # Skip iRODS paths containing any of these substrings.
    skip_substrings = (
        "initial_conditions", "trash",
        "30min", "5min", "oldDownscale_europe",
    )

    for var in variables:
        # Collect the iRODS paths to download for this variable.
        session = setup_iRodsSession(env_config, password)
        with session as s:
            query = s.query(
                Collection.parent_name, Collection.name, DataObject.id, DataObject.name
            ).filter(DataObject.name == var)

            allFiles = []
            for result in query.get_results():
                irods_path = f"{result[Collection.name]}/{var}"  # pure POSIX

                if any(token in irods_path for token in skip_substrings):
                    continue

                # Skip paths that don't match the expected project prefix.
                if not irods_path.startswith(projectDirectory):
                    continue

                allFiles.append(irods_path)

        # Download each file and rewrite it as zarr.
        for server_path in tqdm(allFiles, desc=f'Downloading {var}', disable=False):
            local_path = Path(server_path.replace(projectDirectory, savePath_yoda, 1))

            session = setup_iRodsSession(env_config, password)
            with session as s:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s.data_objects.get(server_path, str(local_path), forceFlag=True)

            ds = xr.open_dataset(local_path, chunks=None, engine='h5netcdf')
            encoding = {name: {"compressors": None} for name in ds.data_vars}
            ds.to_zarr(local_path.with_suffix('.zarr'), mode='w', encoding=encoding, consolidated=False)
            local_path.unlink()

download_from_yoda(env_config, password, data_dir, projectDirectory, variables)
