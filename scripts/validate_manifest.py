import argparse,csv,hashlib,json
from pathlib import Path
from PIL import Image
def validate(file):
    rows=list(csv.DictReader(open(file,encoding='utf-8-sig')));groups={};hashes={};errors=[];ids=set()
    if not rows:errors.append('empty manifest')
    for r in rows:
        rid=r['image_id'];split=r['split'];group=r.get('group_id','')
        if rid in ids:errors.append(f'duplicate ID {rid}')
        ids.add(rid)
        if split not in ['train','val','test']:errors.append(f'{rid}: invalid split')
        if r.get('annotation_complete','').lower()!='true':errors.append(f'{rid}: annotation completeness unverified')
        if r.get('permission_status')!='verified':errors.append(f'{rid}: permissions unverified')
        if group:
            if r.get('group_verified','').lower()!='true':errors.append(f'{rid}: group not verified')
            if group in groups and groups[group]!=split:errors.append(f'group leakage {group}')
            groups[group]=split
        try:
            ip=Path(r['image']);mp=Path(r['mask'])
            with Image.open(ip) as im,Image.open(mp) as ma:
                im.load();ma.load()
                if im.size!=ma.size:errors.append(f'{rid}: dimensions differ')
            h=hashlib.file_digest(ip.open('rb'),'sha256').hexdigest()
            if h in hashes and hashes[h][0]!=split:errors.append(f'duplicate crosses split {rid}/{hashes[h][1]}')
            hashes[h]=(split,rid)
            if split=='test' and r.get('checkpoint_exposure')!='clean':errors.append(f'{rid}: test exposure unresolved')
        except Exception as e:errors.append(f'{rid}: {e}')
    for split in ['train','val','test']:
        if not any(r['split']==split for r in rows):errors.append(f'missing {split}')
    report=dict(rows=len(rows),verified_groups=len(groups),grouping='grouped' if all(r.get('group_id') for r in rows) else 'image_level_or_partial',errors=errors)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if errors:raise ValueError('Manifest fails validation')
    return rows
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('manifest');validate(a.parse_args().manifest)
