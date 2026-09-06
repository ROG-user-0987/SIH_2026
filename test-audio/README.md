# Test Audio Directory

This directory stores test audio samples used for development and demo rehearsals.

## Requirements
- Format: WAV (PCM 16-bit mono preferred)
- Sample rate: 16000 Hz preferred (any rate supported via server-side resampling)
- Duration: 2-30 seconds recommended
- Naming: `<provenance>_<description>.wav` (e.g., `team_recorded_genuine_01.wav`)

## Provenance Tracking
Every file must have a matching `.meta.json`:
```json
{
  "source": "team_recorded | public_dataset | synthetic",
  "consent": true,
  "description": "Brief description",
  "added_by": "Team member name",
  "added_date": "2026-09-06"
}
```

## Privacy Rules (S-09)
- Only team-created synthetic voices or consenting participants
- No third-party recordings without explicit written consent
- No real-person voice clones without consent
