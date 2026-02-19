export function diffDays(dateA, dateB) {
  const msPerDay = 24 * 60 * 60 * 1000;
  const a = new Date(dateA);
  const b = new Date(dateB);
  return Math.round((b - a) / msPerDay);
}

export function jdToIso(jd) {
  const ms = (jd - 2440587.5) * 86400000;
  return new Date(ms).toISOString().split('T')[0];
}

export function isoToJd(iso) {
  const ms = new Date(iso).getTime();
  return ms / 86400000 + 2440587.5;
}

export function jdToUnixMs(jd) {
  // JD 2440587.5 == Unix epoch (1970-01-01T00:00:00Z)
  return (jd - 2440587.5) * 86400 * 1000;
}

export function pointToUnixMs(p) {
  if (p.epoch_jd != null) return jdToUnixMs(Number(p.epoch_jd));
  // If your API uses a different key, add it here.
  return NaN;
}

export function findClosestIndexByTime(points, targetMs) {
  let bestI = 0;
  let bestDiff = Infinity;
  
  for (let i = 0; i < points.length; i++) {
    const t = pointToUnixMs(points[i]);
    if (!Number.isFinite(t)) continue;
    const diff = Math.abs(t - targetMs);

    if (diff < bestDiff) {
      bestDiff = diff;
      bestI = i;
    }
  }

  return bestI;
}
