"""Read-only inventory of local evidence; never promotes predictions to references."""
from pathlib import Path
import csv, json, collections, datetime, zipfile
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(r'D:\Thầy Khánh\NoiSoi')

def main():
    files=[p for p in SOURCE.rglob('*') if p.is_file()]
    families=collections.Counter(); candidates=[]; tables=[]; errors=[]; archives=[]
    for p in files:
        rel=p.relative_to(SOURCE)
        if p.suffix.lower() in {'.zip','.rar','.7z'}:
            if p.suffix.lower()=='.zip':
                try:
                    with zipfile.ZipFile(p) as z:
                        names=z.namelist()
                    archives.append(dict(path=str(rel),members=len(names),
                        reference_candidates=[n for n in names if any(x in n.lower() for x in ['autolaparo','tail_reference','reference_rater','annotation_tasks'])],
                        directory_read=True,payload_crc_checked=False))
                except Exception as exc:errors.append(dict(path=str(rel),error=str(exc)))
            else:archives.append(dict(path=str(rel),directory_read=False,reason='Not a ZIP; contents not inspected by this audit'))
        if p.suffix.lower() in {'.png','.tif','.tiff','.npy','.npz'} and any(x in str(rel).lower() for x in ['mask','annotation','reference','label','ground_truth']):
            families[str(Path(*rel.parts[:2]))]+=1
            candidates.append(str(rel))
        if p.suffix.lower()=='.csv':
            try:
                with p.open(encoding='utf-8-sig',newline='') as f:
                    reader=csv.DictReader(f); fields=reader.fieldnames or []
                    relevant=[k for k in fields if any(x in k.lower() for x in ['mask','polygon','annotation_complete','presence','reviewer','reference_scope','designated_frame'])]
                    rows=list(reader)
                if relevant:
                    tables.append(dict(path=str(rel),rows=len(rows),fields=fields,
                        nonempty={k:sum(bool(str(r.get(k) or '').strip()) for r in rows) for k in relevant}))
            except Exception as exc:errors.append(dict(path=str(rel),error=str(exc)))
    report=dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(SOURCE),
        total_files=len(files),candidate_mask_artifact_families=dict(families),
        candidate_paths=candidates,reference_related_tables=tables,archives=archives,read_errors=errors,
        conclusion='No tail reference set approved by this inventory. Candidate artifacts require source/frame/scope/completeness verification; blank localization reader tables are not binary segmentation masks.',
        source_modified=False)
    out=ROOT/'materials/verification/noisoi_reference_search_20261016.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(total_files=len(files),candidate_families=dict(families),reference_tables=len(tables),archives=len(archives),read_errors=errors),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
