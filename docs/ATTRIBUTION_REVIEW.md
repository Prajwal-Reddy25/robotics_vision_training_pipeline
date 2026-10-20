# Photo attribution review — 2026-10-20

The original [credits](evidence/v1_2/gallery/credits.json),
[attribution text](evidence/v1_2/gallery/ATTRIBUTION.md),
[selection](evidence/v1_2/gallery/selection.json) and PNGs remain byte-for-byte unchanged.
New read-only public Flickr oEmbed queries are recorded in
[photo_links.json](reviews/2026-10-20/photo_links.json).

All **nine recorded creators** resolve to the same public creator name, author URL
and photo title. Their responses are HTTP 200 and type `photo`. The two README
figures are COCO 206218 (Mechatronics Guy) and COCO 554291 (Eirik Newth); both have
recorded source-metadata CC BY 2.0, creator/source links and resizing modifications.
Ground-truth maps credit the COCO Consortium under CC BY 4.0.

| COCO ID | Recorded creator | Current public metadata |
| --- | --- | --- |
| 68765 | DeclanTM | matches |
| 206218 | Mechatronics Guy | matches |
| 256916 | @joefoodie | matches |
| 333772 | mikes rite | matches |
| 343466 | SJU Undergraduate Admissions | matches |
| 383842 | Chairman of the Joint Chiefs of Staff | matches |
| 458410 | amy_elizabeth_west | matches |
| 498919 | santanartist | matches |
| 554291 | Eirik Newth | matches |

## Unresolved cases

- **287347:** oEmbed returns HTTP 200, type `link`, with no creator/title. A successful
  HTTP response alone does not resolve attribution. Original photograph remains
  omitted from the tracked diagnostic panel.
- **464358:** oEmbed returns HTTP 404. Creator remains unresolved; its original
  photograph remains omitted from the tracked panel.

Neither case is added to the README gallery, assigned an invented author, or removed
from numerical evaluation. Ground-truth/prediction diagnostics remain available.
The historical `http://farm*.staticflickr.com` URLs are retained as provenance,
not substituted into the README or silently rewritten in the frozen metadata.

This is a current public-metadata/source-link check, not an independent legal rights
audit or a claim that metadata license 4 certifies all current rights. Photo-license
selection comes from the released COCO metadata; oEmbed does not re-establish the
license. External links may later change. No request to another person, photo upload
or publication was made.
