// rpchain <roadpaint out dir>: world-space paint polylines from roadpaint's exact segments.
//
// Keeps the segments the tiles show as painted (another surface or an opaque decal can cover a texture's line),
// joins pieces that share endpoints into continuous paint, joins those end to end across dash gaps (straight on, up
// to 15 m) into lines, and writes polylines.jsonl, one line per record:
//   {"id", "colour": white|yellow, "style": solid|dashed, "width" m, "len" m, "painted" fraction, "src": surface|decal,
//    "pair": id of the parallel line of a double line or null, "pts": [[x,y,z]...] (2 cm simplified),
//    "dashes": [[s0,s1]...] painted stretches as distances along pts (dashed lines only)}
// Crossing textures' stripes are left out (they are crossings, in features). Env RP_BOX="x0 y0 x1 y1" limits the area.
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Numerics;
using System.Text;

var dir = args[0];
var inv = System.Globalization.CultureInfo.InvariantCulture;
var sw = System.Diagnostics.Stopwatch.StartNew();
void Log(string s) => Console.Error.WriteLine($"[{sw.Elapsed:hh\\:mm\\:ss}] {s}");
float[] box = (Environment.GetEnvironmentVariable("RP_BOX") ?? "-1e9 -1e9 1e9 1e9").Split(' ').Select(s => float.Parse(s, inv)).ToArray();

var texName = new Dictionary<int, string>();
foreach (var line in File.ReadLines(Path.Combine(dir, "textures.tsv")).Skip(1)) { var f = line.Split('\t'); texName[int.Parse(f[0])] = f[1]; }

// segments
var raw = File.ReadAllBytes(Path.Combine(dir, "segments.bin"));
const int RS = 7 * 4 + 1 + 1 + 2 + 4;
int nRaw = raw.Length / RS;
var segs = new List<Seg>(nRaw / 2);
for (int i = 0; i < nRaw; i++)
{
  int o = i * RS;
  var s = new Seg
  {
    A = new Vector3(BitConverter.ToSingle(raw, o), BitConverter.ToSingle(raw, o + 4), BitConverter.ToSingle(raw, o + 8)),
    B = new Vector3(BitConverter.ToSingle(raw, o + 12), BitConverter.ToSingle(raw, o + 16), BitConverter.ToSingle(raw, o + 20)),
    W = BitConverter.ToSingle(raw, o + 24), Colour = raw[o + 28], Decal = raw[o + 29] == 1, Tex = BitConverter.ToUInt16(raw, o + 30),
  };
  if (s.W < 0.03f || s.W > 1.2f || (s.B - s.A).Length() < 1e-3f) continue;
  if (s.A.X < box[0] || s.A.X > box[2] || s.A.Y < box[1] || s.A.Y > box[3]) continue;
  if (texName.TryGetValue(s.Tex, out var tn) && tn.Contains("crossing")) continue;
  segs.Add(s);
}
raw = null;
Log($"{nRaw} segments, {segs.Count} kept by width/area");

// visibility against the tiles: the painted colour at the segment's height within 1 px, at 1/4, 1/2 and 3/4
const float TILE = 64f, RES = 0.05f; const int N = 1280;
var tileDir = Path.Combine(dir, "tiles");
(byte[] code, float[] z, int nl)? cur = null; long curKey = long.MinValue;
(byte[], float[], int)? LoadTile(int tx, int ty)
{
  var p = Path.Combine(tileDir, $"{tx}_{ty}.bin");
  if (!File.Exists(p)) return null;
  var b = File.ReadAllBytes(p);
  int nl = BitConverter.ToInt32(b, 20);
  int o = 24;
  byte[] Block(ref int off) { int len = BitConverter.ToInt32(b, off); off += 4; using var ms = new MemoryStream(b, off, len); using var zs = new ZLibStream(ms, CompressionMode.Decompress); using var outp = new MemoryStream(); zs.CopyTo(outp); off += len; return outp.ToArray(); }
  var code = Block(ref o);
  var zb = Block(ref o);
  var z = new float[zb.Length / 4]; Buffer.BlockCopy(zb, 0, z, 0, zb.Length);
  return (code, z, nl);
}
long TileKey(float x, float y) => ((long)Math.Floor(x / TILE) << 32) ^ (uint)(int)Math.Floor(y / TILE);
segs.Sort((a, b) => TileKey((a.A.X + a.B.X) / 2, (a.A.Y + a.B.Y) / 2).CompareTo(TileKey((b.A.X + b.B.X) / 2, (b.A.Y + b.B.Y) / 2)));
var tileCache = new Dictionary<long, (byte[], float[], int)?>();
bool Painted(Vector3 p, byte colour)
{
  long key = TileKey(p.X, p.Y);
  if (!tileCache.TryGetValue(key, out var t))
  {
    if (tileCache.Count > 12) tileCache.Clear();
    t = LoadTile((int)Math.Floor(p.X / TILE), (int)Math.Floor(p.Y / TILE));
    tileCache[key] = t;
  }
  if (t == null) return false;
  var (code, z, nl) = t.Value;
  float x0 = (float)Math.Floor(p.X / TILE) * TILE, y1 = (float)Math.Floor(p.Y / TILE) * TILE + TILE;
  int c = (int)((p.X - x0) / RES), r = (int)((y1 - p.Y) / RES);
  for (int dr = -1; dr <= 1; dr++)
    for (int dc = -1; dc <= 1; dc++)
    {
      int rr = r + dr, cc = c + dc;
      if (rr < 0 || cc < 0 || rr >= N || cc >= N) continue;
      for (int l = 0; l < nl; l++)
      {
        int i = l * N * N + rr * N + cc;
        if ((code[i] >> 3) == colour && Math.Abs(z[i] - p.Z) < 0.6f) return true;
      }
    }
  return false;
}
var vis = new List<Seg>(segs.Count);
foreach (var s in segs)
{
  int n = 0, ok = 0;
  foreach (var f in (s.B - s.A).Length() < 0.3f ? new[] { 0.5f } : new[] { 0.25f, 0.5f, 0.75f })
  { n++; if (Painted(s.A + (s.B - s.A) * f, s.Colour)) ok++; }
  if (ok * 2 > n) vis.Add(s);
}
segs = vis;
tileCache.Clear();
Log($"{segs.Count} visible");

// join pieces sharing endpoints (within 1.5 cm, same colour) into runs of continuous paint
var nodes = new List<Vector3>(); var nodeCol = new List<byte>();
var grid = new Dictionary<(int, int), List<int>>();
int Node(Vector3 p, byte colour)
{
  int gx = (int)Math.Floor(p.X / 0.02f), gy = (int)Math.Floor(p.Y / 0.02f);
  for (int dx = -1; dx <= 1; dx++)
    for (int dy = -1; dy <= 1; dy++)
      if (grid.TryGetValue((gx + dx, gy + dy), out var l))
        foreach (var id in l)
          if (nodeCol[id] == colour && Vector2.Distance(new Vector2(nodes[id].X, nodes[id].Y), new Vector2(p.X, p.Y)) < 0.015f && Math.Abs(nodes[id].Z - p.Z) < 0.5f) return id;
  nodes.Add(p); nodeCol.Add(colour);
  if (!grid.TryGetValue((gx, gy), out var g)) grid[(gx, gy)] = g = new List<int>();
  g.Add(nodes.Count - 1);
  return nodes.Count - 1;
}
var adj = new Dictionary<int, List<int>>();  // node -> segment indices
var edgeSeen = new HashSet<(int, int)>();
var segA = new List<int>(); var segB = new List<int>(); var segRef = new List<Seg>();
foreach (var s in segs)
{
  int a = Node(s.A, s.Colour), b = Node(s.B, s.Colour);
  if (a == b || !edgeSeen.Add((Math.Min(a, b), Math.Max(a, b)))) continue; // overlapping copies of one piece
  int k = segRef.Count;
  segA.Add(a); segB.Add(b); segRef.Add(s);
  if (!adj.TryGetValue(a, out var la)) adj[a] = la = new List<int>(); la.Add(k);
  if (!adj.TryGetValue(b, out var lb)) adj[b] = lb = new List<int>(); lb.Add(k);
}
grid = null;
Log($"{segRef.Count} pieces, {nodes.Count} nodes");
var used = new bool[segRef.Count];
var runs = new List<Run>();
void Walk(int start, int firstSeg)
{
  var pts = new List<int> { start };
  var ws = new List<float>(); int nDecal = 0;
  int node = start, sg = firstSeg;
  while (sg >= 0 && !used[sg])
  {
    used[sg] = true;
    ws.Add(segRef[sg].W); if (segRef[sg].Decal) nDecal++;
    node = segA[sg] == node ? segB[sg] : segA[sg];
    pts.Add(node);
    var l = adj[node];
    sg = l.Count == 2 ? (l[0] == sg ? l[1] : l[0]) : -1;
  }
  ws.Sort();
  runs.Add(new Run { Pts = pts.Select(i => nodes[i]).ToList(), Colour = nodeCol[start], W = ws[ws.Count / 2], DecalFrac = nDecal / (float)ws.Count });
}
foreach (var (n, l) in adj) if (l.Count != 2) foreach (var sg in l) if (!used[sg]) Walk(n, sg);
for (int k = 0; k < segRef.Count; k++) if (!used[k]) Walk(segA[k], k);  // loops
Log($"{runs.Count} runs");

// join runs end to end across dash gaps: on (within 25 deg, sideways 0.15 m + 5% of the gap for curves), up to 15 m, same colour and
// a similar width; each run end joins at most once, nearest first
foreach (var r in runs) r.Simplify(0.02f);
var ends = new Dictionary<(int, int), List<(int run, int end)>>();
for (int i = 0; i < runs.Count; i++)
  for (int e = 0; e < 2; e++)
  {
    var p = runs[i].End(e);
    var key = ((int)Math.Floor(p.X / 16), (int)Math.Floor(p.Y / 16));
    if (!ends.TryGetValue(key, out var l)) ends[key] = l = new List<(int, int)>();
    l.Add((i, e));
  }
var cand = new List<(float d, int ra, int ea, int rb, int eb)>();
for (int i = 0; i < runs.Count; i++)
  for (int e = 0; e < 2; e++)
  {
    var R = runs[i];
    if (R.Length() < 0.3f) continue;
    var p = R.End(e); var d = R.OutDir(e);
    int gx = (int)Math.Floor(p.X / 16), gy = (int)Math.Floor(p.Y / 16);
    for (int dx = -1; dx <= 1; dx++)
      for (int dy = -1; dy <= 1; dy++)
        if (ends.TryGetValue((gx + dx, gy + dy), out var l))
          foreach (var (j, f) in l)
          {
            if (j <= i) continue;
            var S = runs[j];
            if (S.Colour != R.Colour || S.Length() < 0.3f || Math.Max(S.W, R.W) > 1.6f * Math.Min(S.W, R.W)) continue;
            var q = S.End(f); var dq = S.OutDir(f);
            var g = q - p;
            float ahead = Vector2.Dot(new Vector2(g.X, g.Y), d);
            float side = Math.Abs(d.X * g.Y - d.Y * g.X);
            if (ahead < 0.05f || ahead > 15f || side > 0.15f + 0.05f * ahead || Math.Abs(g.Z) > 1f) continue;  // curves bend away from the straight extension
            if (Vector2.Dot(d, dq) > -0.906f) continue; // the other run must head back at us: within 25 degrees
            cand.Add((ahead, i, e, j, f));
          }
  }
cand.Sort((a, b) => a.d.CompareTo(b.d));
var link = new Dictionary<(int, int), (int, int)>();
foreach (var c in cand)
{
  if (link.ContainsKey((c.ra, c.ea)) || link.ContainsKey((c.rb, c.eb))) continue;
  link[(c.ra, c.ea)] = (c.rb, c.eb); link[(c.rb, c.eb)] = (c.ra, c.ea);
}
// assemble lines: walk from runs whose one end is free
var inLine = new bool[runs.Count];
var lines = new List<Line>();
void Assemble(int start, int freeEnd)
{
  var L = new Line { Colour = runs[start].Colour };
  int r = start, entry = freeEnd;
  while (true)
  {
    inLine[r] = true;
    var pts = runs[r].Pts.ToList();
    if (entry == 1) pts.Reverse();
    L.Add(pts, runs[r].W, runs[r].DecalFrac);
    int exit = 1 - entry;
    if (!link.TryGetValue((r, exit), out var nx) || inLine[nx.Item1]) break;
    r = nx.Item1; entry = nx.Item2;
  }
  lines.Add(L);
}
for (int i = 0; i < runs.Count; i++) if (!inLine[i]) { if (!link.ContainsKey((i, 0))) Assemble(i, 0); else if (!link.ContainsKey((i, 1))) Assemble(i, 1); }
for (int i = 0; i < runs.Count; i++) if (!inLine[i]) Assemble(i, 0); // closed loops of dashes
lines.RemoveAll(l => l.Len < 0.6f || l.W < 0.07f);  // texture specks and cracks, not paint (lines are 0.1-0.3 m wide)
Log($"{lines.Count} lines");

// double lines: a same-coloured line 0.08-0.45 m to the side along at least half of the shorter one
var sgrid = new Dictionary<(int, int), List<(int line, Vector3 p, Vector2 d)>>();
for (int i = 0; i < lines.Count; i++)
  foreach (var (p, d) in lines[i].Samples(1f))
  {
    var key = ((int)Math.Floor(p.X / 2), (int)Math.Floor(p.Y / 2));
    if (!sgrid.TryGetValue(key, out var l)) sgrid[key] = l = new List<(int, Vector3, Vector2)>();
    l.Add((i, p, d));
  }
for (int i = 0; i < lines.Count; i++)
{
  var votes = new Dictionary<int, int>(); int n = 0;
  foreach (var (p, d) in lines[i].Samples(1f))
  {
    n++;
    var hit = new HashSet<int>();
    int gx = (int)Math.Floor(p.X / 2), gy = (int)Math.Floor(p.Y / 2);
    for (int dx = -1; dx <= 1; dx++)
      for (int dy = -1; dy <= 1; dy++)
        if (sgrid.TryGetValue((gx + dx, gy + dy), out var l))
          foreach (var (j, q, dq) in l)
          {
            if (j == i || lines[j].Colour != lines[i].Colour || hit.Contains(j)) continue;
            var g = q - p;
            float side = Math.Abs(d.X * g.Y - d.Y * g.X), ahead = Math.Abs(Vector2.Dot(new Vector2(g.X, g.Y), d));
            if (side > 0.08f && side < 0.45f && ahead < 0.6f && Math.Abs(Vector2.Dot(d, dq)) > 0.97f && Math.Abs(g.Z) < 0.5f) hit.Add(j);
          }
    foreach (var j in hit) votes[j] = votes.GetValueOrDefault(j) + 1;
  }
  int best = -1, bv = 0;
  foreach (var (j, v) in votes) if (v > bv) { best = j; bv = v; }
  if (best >= 0 && bv >= 0.5f * Math.Min(n, lines[best].Samples(1f).Count())) lines[i].Pair = best;
}

using (var w = new StreamWriter(Path.Combine(dir, "polylines.jsonl")))
  for (int i = 0; i < lines.Count; i++)
  {
    var L = lines[i];
    float painted = L.Dashes.Sum(d => d.b - d.a) / L.Len;
    bool dashed = L.Dashes.Count > 1 && L.MaxGap() > 1.0f;
    var sb = new StringBuilder();
    sb.Append(string.Format(inv, "{{\"id\":{0},\"colour\":\"{1}\",\"style\":\"{2}\",\"width\":{3:F2},\"len\":{4:F2},\"painted\":{5:F2},\"src\":\"{6}\",\"pair\":{7},\"pts\":[",
      i, L.Colour == 2 ? "yellow" : "white", dashed ? "dashed" : "solid", L.W, L.Len, painted, L.DecalFrac > 0.5f ? "decal" : "surface", L.Pair >= 0 ? L.Pair.ToString() : "null"));
    sb.Append(string.Join(",", L.Pts.Select(p => string.Format(inv, "[{0:F2},{1:F2},{2:F2}]", p.X, p.Y, p.Z))));
    sb.Append("]");
    if (dashed) sb.Append(",\"dashes\":[").Append(string.Join(",", L.Dashes.Select(d => string.Format(inv, "[{0:F2},{1:F2}]", d.a, d.b)))).Append("]");
    sb.Append("}");
    w.WriteLine(sb.ToString());
  }
Log("done");
return 0;

public class Seg { public Vector3 A, B; public float W; public byte Colour; public bool Decal; public ushort Tex; }

public class Run
{
  public List<Vector3> Pts; public byte Colour; public float W, DecalFrac;
  public Vector3 End(int e) => e == 0 ? Pts[0] : Pts[^1];
  public float Length() { float s = 0; for (int i = 1; i < Pts.Count; i++) s += (Pts[i] - Pts[i - 1]).Length(); return s; }
  // outward direction at an end, over its last ~1 m
  public Vector2 OutDir(int e)
  {
    var p = End(e); Vector3 q = p; float acc = 0;
    for (int k = 1; k < Pts.Count; k++)
    {
      var a = e == 0 ? Pts[k] : Pts[^(k + 1)];
      q = a; acc = (new Vector2(a.X - p.X, a.Y - p.Y)).Length();
      if (acc > 1f) break;
    }
    var d = new Vector2(p.X - q.X, p.Y - q.Y);
    return d.Length() < 1e-6f ? Vector2.UnitX : Vector2.Normalize(d);
  }
  public void Simplify(float tol) { Pts = Line.DouglasPeucker(Pts, tol); }
}

public class Line
{
  public List<Vector3> Pts = new(); public List<(float a, float b)> Dashes = new();
  public byte Colour; public float W, DecalFrac; public int Pair = -1;
  float wsum, lsum, dsum;
  public float Len => Pts.Count < 2 ? 0 : Dist(Pts.Count - 1);
  List<float> cum = new() ;
  float Dist(int i) => cum[i];
  public void Add(List<Vector3> pts, float w, float decal)
  {
    int first = Pts.Count;
    foreach (var p in pts)
    {
      cum.Add(Pts.Count == 0 ? 0 : cum[^1] + (p - Pts[^1]).Length());  // a dash gap counts as its straight distance
      Pts.Add(p);
    }
    float a = cum[first];
    Dashes.Add((a, cum[^1]));
    float runLen = cum[^1] - a;
    wsum += w * runLen; dsum += decal * runLen; lsum += runLen;
    W = lsum > 0 ? wsum / lsum : w; DecalFrac = lsum > 0 ? dsum / lsum : decal;
  }
  public float MaxGap() { float g = 0; for (int i = 1; i < Dashes.Count; i++) g = Math.Max(g, Dashes[i].a - Dashes[i - 1].b); return g; }
  public IEnumerable<(Vector3, Vector2)> Samples(float step)
  {
    for (int i = 1; i < Pts.Count; i++)
    {
      var a = Pts[i - 1]; var b = Pts[i];
      var d2 = new Vector2(b.X - a.X, b.Y - a.Y); float l = d2.Length();
      if (l < 1e-4f) continue;
      d2 /= l;
      for (float s = 0; s < l; s += step) yield return (a + (b - a) * (s / l), d2);
    }
  }
  public static List<Vector3> DouglasPeucker(List<Vector3> p, float tol)
  {
    if (p.Count < 3) return p;
    var keep = new bool[p.Count]; keep[0] = keep[^1] = true;
    var st = new Stack<(int, int)>(); st.Push((0, p.Count - 1));
    while (st.Count > 0)
    {
      var (a, b) = st.Pop();
      float best = 0; int bi = -1;
      var ab = p[b] - p[a]; float l2 = ab.LengthSquared();
      for (int i = a + 1; i < b; i++)
      {
        var ap = p[i] - p[a];
        float t = l2 > 0 ? Math.Clamp(Vector3.Dot(ap, ab) / l2, 0, 1) : 0;
        float d = (ap - ab * t).Length();
        if (d > best) { best = d; bi = i; }
      }
      if (bi >= 0 && best > tol) { keep[bi] = true; st.Push((a, bi)); st.Push((bi, b)); }
    }
    return p.Where((_, i) => keep[i]).ToList();
  }
}
