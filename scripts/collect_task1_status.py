"""Fetch small integrity/probe receipts only; never download Task 1 videos locally."""
from pathlib import Path
import json,datetime,concurrent.futures
import modal
ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'materials/verification/task1_mirror_manifest.json').read_text())
    volume=modal.Volume.from_name('isbi-gynsurg-task1-recovery-v1',create_if_missing=False)
    out=ROOT/'materials/verification/task1_receipts_v1';out.mkdir(exist_ok=True)
    receipts=[];probes={}
    entries=[e for e in volume.iterdir('/'+manifest['revision'],recursive=True) if Path(e.path).name in ['download_verified.json','ffprobe.json','recovery_report.json']]
    def fetch(entry):
        name=Path(entry.path).name
        text=b''.join(volume.read_file(entry.path)).decode();row=json.loads(text)
        vid=Path(entry.path).parent.name
        (out/(vid+'_'+name)).write_text(text,encoding='utf8')
        return vid,name,row
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:results=list(ex.map(fetch,entries))
    for vid,name,row in results:
        if name=='download_verified.json':receipts.append(row)
        elif name=='ffprobe.json':probes[vid]=row
    labels={r['video_id']:r for r in json.loads((ROOT/'materials/verification/task1_original_label_inventory.json').read_text())}
    alignment=[]
    for vid,p in sorted(probes.items()):
        s=next(x for x in p['streams'] if x['codec_type']=='video');duration=float(p['format']['duration'])
        alignment.append(dict(video_id=vid,video_duration_s=duration,label_rows=labels[vid]['label_rows'],
            difference_s=duration-labels[vid]['label_rows'],fps=s['r_frame_rate'],width=s['width'],height=s['height']))
    status=dict(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        repo=manifest['repo'],revision=manifest['revision'],expected_videos=21,
        expected_bytes=manifest['total_bytes'],downloaded_sha256_verified_videos=len(receipts),
        downloaded_sha256_verified_bytes=sum(r['bytes'] for r in receipts),
        indexed_videos=len(list((ROOT/'materials/verification/task1_recovery_v1').glob('??.json'))),
        raw_storage='Modal CONFIGURED_MODAL_PROFILE / isbi-gynsurg-task1-recovery-v1',
        all_task1_videos_downloaded=len(receipts)==21,original_label_duration_checks=alignment,
        mapping_verified=(json.loads((ROOT/'configs/active_sequence_review.json').read_text()).get('grouping_verified',False) if (ROOT/'configs/active_sequence_review.json').exists() else False),primary_evaluation_complete=False)
    (ROOT/'materials/verification/task1_recovery_status.json').write_text(json.dumps(status,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in status.items() if k!='original_label_duration_checks'},indent=2))

if __name__=='__main__':main()
