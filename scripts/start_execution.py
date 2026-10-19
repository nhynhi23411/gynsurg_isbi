"""Record the user's explicit authorization; do not fabricate data/ethics approvals."""
from pathlib import Path
import hashlib,json,datetime
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'protocols/20261016_execution_v1';out.mkdir(parents=True,exist_ok=True)
plan=(ROOT/'PLAN_RULES.md').read_bytes();snapshot=out/'PLAN_RULES_APPROVED_SCOPE.md'
if snapshot.exists() and snapshot.read_bytes()!=plan:raise RuntimeError('Do not overwrite an approved snapshot')
snapshot.write_bytes(plan);fingerprint=hashlib.sha256(plan).hexdigest()
record={'authorized_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization_source':'Direct user messages in this chat on 2026-10-16','request':'thực hiện đầu tiên full đầy đủ với isbi gynssurg trước, đáp ứng đúng plan + rule','modal_authorization':'momodal lấy trên modal đi; tôi có 5 account modal, cứ coi acc nào còn credit mà dùng, cứ dùng','protocol_sha256':fingerprint,'scope':'ISBI_GynSurg execution; existing Modal profiles/volumes and GPU compute; independent versioned outputs','limits':['No fabricated human labels, procedure IDs, ethics approvals or remaining credit balances','No final-test tuning; freeze protocol and verify scientific gates before final evaluation','No modification of old research/raw hardlinks; no paper submission or third-party messaging'],'data_gates_not_waived':True}
(out/'USER_AUTHORIZATION.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
approval={'approved_for_training':True,'approved_for_test':False,'reviewer':'user — direct instruction in this chat','date':'2026-10-16','protocol_sha256':fingerprint,'authorization_record':str((out/'USER_AUTHORIZATION.json').relative_to(ROOT)),'note':'Development execution authorized. Grouping, reference completeness, permission and exposure remain independent prerequisites. Test access enabled only after protocol freeze.'}
(ROOT/'configs/review_status.json').write_text(json.dumps(approval,ensure_ascii=False,indent=2),encoding='utf8')
print('User authorization recorded; development enabled, final test still locked')
