"""Update protocol wording to match the versioned no-new-label amendment."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=ROOT/'paper/main.tex';t=p.read_text(encoding='utf8')
    t=t.replace('Source grouping and additional human references are being completed.','Source grouping is being verified, and the computational protocol uses existing official references without new annotation.')
    t=t.replace('The proposed targets are uterus and the union of annotated instrument parts, subject to clinical scope and completeness review. Missing annotations are never converted to absence.','The computational targets are uterus and the union of annotated instrument parts, using the dataset-defined reference scope. Engineering checks of encoding, alignment and sample overlays do not constitute a new clinical boundary review. Missing annotations are never converted to absence.')
    t=t.replace('The operational draft uses query frames $[0,24,49,74,99]$ and scoring frames $[124,149,174,199,224]$. The four tail frames require new human masks. An annotation package contains 1,200 images and 2,400 unfilled structure tasks; it is not a completed reference set. Clinical annotation conventions and subset adjudication remain pending.',
      'A versioned amendment uses only the first 125 raw frames. Query frames are $[0,24,49]$ and scoring frames are $[74,99,124]$, shared by every policy. All six references are supplied by Task~3; no tail masks or new annotations are required for this narrower benchmark. The query window precedes the scoring window, so conclusions are restricted to later annotated snapshots in this short window. No independent clinical review or inter-reader validation is claimed.')
    t=t.replace('With $T=250$ and ten candidates, counts are $[0,2,5,10]$ and actual densities $[0,0.8,2,4]$.','With $T=125$ and six candidates, counts are $[0,1,2,5]$ and actual densities $[0,0.8,1.6,4]$.')
    t=t.replace('Institution-specific confirmation, new-annotation arrangements and permissions for published examples remain to be verified.','Institution-specific confirmation and permissions for published examples remain to be verified. The amended benchmark creates no new human annotations.')
    p.write_text(t,encoding='utf8')
    b=ROOT/'paper/refs.bib';text=b.read_text(encoding='utf8')
    # IEEEbib does not render the nonstandard eprint field; retain explicit identifiers.
    if 'howpublished={arXiv:' not in text:
        text=re.sub(r'\n      eprint=\{([^}]+)\},',lambda m:'\n      howpublished={arXiv:'+m.group(1)+'},\n      eprint={'+m.group(1)+'},',text)
    b.write_text(text,encoding='utf8')
    print('Updated no-new-label protocol wording and printable arXiv identifiers; results remain pending')
if __name__=='__main__':main()
