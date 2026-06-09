import os
import subprocess

adjust_file_list = ['results', 'results_marmousi']
optimizer_list = ['0.01Muon']

def speedup_video(input_path, output_path):
    if not os.path.exists(input_path):
        return False
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-filter:v', 'setpts=0.4*PTS',
        '-an', output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False

def main():
    for root in adjust_file_list:
        for opt in optimizer_list:
            in_path = os.path.join(root, opt, 'model_eps_video.mp4')
            out_path = os.path.join(root, opt, 'model_eps_video_2.5x.mp4')
            ok = speedup_video(in_path, out_path)
            print(f'{root}/{opt}: {"done" if ok else "skipped"}')

if __name__ == '__main__':
    main()
