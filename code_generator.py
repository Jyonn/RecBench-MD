import os
import shlex
import subprocess

import pigmento
from pigmento import pnt

from utils.config_init import ConfigInit


def run_script(command, directory):
    args = shlex.split(command)
    pnt(command)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=directory)

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    process.stdout.close()
    return process.wait()


if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    configuration = ConfigInit(
        required_args=['data', 'model'],
        default_args=dict(
            seq=False,
            pca=False,
            rqvae_package='../RQ-VAE'
        ),
        makedirs=[]
    ).parse()

    data, model = configuration.data, configuration.model
    rqvae = configuration.rqvae_package

    # detect whether rqvae exists using os
    if not os.path.exists(rqvae):
        raise ValueError(f'rqvae package not found: {rqvae}')

    type_ = '.seq' if configuration.seq else ''
    embed_path = f'../BenchLLM4RS/export/{data}/{model}-embeds{type_}.npy'
    if configuration.pca:
        embed_path = embed_path.replace('.npy', f'-pca{configuration.pca}.npy')

    rqvae_ckpt_path = f'./ckpt/{data}.{model}{type_}'

    pnt(f'learning code for {data} using {model} embeddings')
    return_code = run_script(
        f'python main.py '
        f'--data_path {embed_path} '
        f'--ckpt_dir {rqvae_ckpt_path} '
        f'--verbose 0',
        directory=rqvae
    )
    assert return_code == 0, 'rqvae training failed'

    pnt(f'generating code for {data} using {model} embeddings')
    return_code = run_script(
        f'python generate_indices_distance.py '
        f'--ckpt_path ./ckpt/{data}.{model}{type_}/best_collision_model.pth '
        f'--output_dir ./ckpt '
        f'--output_file {data}.{model}{type_}.json '
        f'--verbose 0',
        directory=rqvae
    )
    assert return_code == 0, 'code generation failed'

    os.makedirs('code', exist_ok=True)

    pnt(f'converting code for {data} using {model} embeddings into RecBench format')
    run_script(
        f'python rq_coder.py '
        f'--data {data} '
        f'--rq {rqvae}/ckpt/{data}.{model}{type_}.json '
        f'--export ./code/{data}.{model}{type_}.code '
        f'--seq {int(configuration.seq)}',
        directory='.'
    )
    assert return_code == 0, 'code conversion failed'
