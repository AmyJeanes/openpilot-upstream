// terrainprobe <game> <out.jsonl> <spec>...: which terrain texture layers the visible ground is painted with at given
// points, straight from the game's archives (only reads them).
//
// Specs:
//   xs:<label>:x,y;x,y;...        a path: one cross-section at each vertex, perpendicular to the path there
//   pt:<label>:x,y                a point with no direction: an east-west and a north-south cross-section through it
//                                 (s grows to the east and to the north)
//   box:<label>:x0,y0,x1,y1,step  a grid of samples
//   file:<label>:<path>           one sample per line of a text file: x y (spaces, tabs or commas between)
// Env: TP_HALF (cross-section half length, default 15 m), TP_STEP (default 0.5 m), TP_QUIET=1 (no tables on stdout),
//      TP_SURVEY=<file.tsv> (also list every terrain geometry in the map: entity, shader, bounds, plan area and each
//      layer texture's share of that area; with no specs, only this).
//
// Every HD map entity's drawable geometry over a sample is a "hit". For the terrain shaders (terrain_cb_*) the hit lists
// the four layer textures with their blend weights, worked out as CodeWalker's TerrainPS.hlsl does:
//   blend = lookup texture at texcoord 1 (if the shader has one), mixed with vertex colour 1 by vertex colour 0's alpha
//           (the _cm shaders have no vertex colour 1: the lookup texture alone)
//   weights of layers 0..3 = (1-b)(1-g), b(1-g), (1-b)g, bg   with b, g the blend's blue and green
// Other shaders list their diffuse texture, so decals and road meshes lying on the terrain show up too.
// One JSON line per sample: {"label","s","x","y","hits":[{"z","ent","shader","kind","layers":[{"tex","w"}],...}]}, hits
// from the top down. Stdout gets a table per cross-section: the top terrain hit's layers and anything lying on it.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Xml;
using CodeWalker;
using CodeWalker.GameFiles;
using SharpDX;

var inv = System.Globalization.CultureInfo.InvariantCulture;
var game = args[0]; var outPath = args[1];
float half = float.Parse(Environment.GetEnvironmentVariable("TP_HALF") ?? "15", inv), step = float.Parse(Environment.GetEnvironmentVariable("TP_STEP") ?? "0.5", inv);
bool quiet = Environment.GetEnvironmentVariable("TP_QUIET") == "1";
var sw = System.Diagnostics.Stopwatch.StartNew();
void Log(string s) => Console.Error.WriteLine($"[{sw.Elapsed:hh\\:mm\\:ss}] {s}");

var samples = new List<Sample>();
void Section(string label, float cx, float cy, float dx, float dy)
{
  float l = MathF.Sqrt(dx * dx + dy * dy); float nx = dy / l, ny = -dx / l;  // to the right of the direction of travel
  for (float s = -half; s <= half + 1e-4f; s += step) samples.Add(new Sample { Label = label, S = s, X = cx + nx * s, Y = cy + ny * s });
}
foreach (var spec in args.Skip(2))
{
  var f = spec.Split(':', 3);
  if (f[0] == "file")
  {
    foreach (var line in File.ReadLines(f[2]))
    {
      var q = line.Split(new[] { ' ', '\t', ',' }, StringSplitOptions.RemoveEmptyEntries);
      if (q.Length >= 2) samples.Add(new Sample { Label = f[1], S = float.NaN, X = float.Parse(q[0], inv), Y = float.Parse(q[1], inv) });
    }
    continue;
  }
  var pts = f[2].Split(';').Select(q => q.Split(',').Select(v => float.Parse(v, inv)).ToArray()).ToArray();
  if (f[0] == "xs")
    for (int i = 0; i < pts.Length; i++)
    {
      var a = pts[Math.Max(i - 1, 0)]; var b = pts[Math.Min(i + 1, pts.Length - 1)];
      Section($"{f[1]}/{i}", pts[i][0], pts[i][1], b[0] - a[0], b[1] - a[1]);
    }
  else if (f[0] == "pt") { Section($"{f[1]}/ew", pts[0][0], pts[0][1], 0, 1); Section($"{f[1]}/ns", pts[0][0], pts[0][1], -1, 0); }
  else if (f[0] == "box")
  {
    var p = pts[0];
    for (float y = p[1]; y <= p[3] + 1e-4f; y += p[4])
      for (float x = p[0]; x <= p[2] + 1e-4f; x += p[4]) samples.Add(new Sample { Label = f[1], S = float.NaN, X = x, Y = y });
  }
  else throw new ArgumentException(spec);
}
var surveyPath = Environment.GetEnvironmentVariable("TP_SURVEY");
using var survey = surveyPath != null ? new StreamWriter(surveyPath) : null;
survey?.WriteLine("ent\tshader\tx0\ty0\tx1\ty1\tarea\ttris\tlookup\tlayers");
float bx0 = float.MaxValue, bx1 = float.MinValue, by0 = float.MaxValue, by1 = float.MinValue;
if (samples.Count > 0) (bx0, bx1, by0, by1) = (samples.Min(s => s.X), samples.Max(s => s.X), samples.Min(s => s.Y), samples.Max(s => s.Y));
float ex0 = bx0, ex1 = bx1, ey0 = by0, ey1 = by1;  // entities to read: those over the samples, or all for a survey
if (survey != null) (ex0, ex1, ey0, ey1) = (-1e9f, 1e9f, -1e9f, 1e9f);
const float CELL = 8f;
var cells = new Dictionary<long, List<int>>();
long CellKey(int cx, int cy) => ((long)(cx + 100000) << 24) | (long)(cy + 100000);
for (int i = 0; i < samples.Count; i++)
{
  var k = CellKey((int)MathF.Floor(samples[i].X / CELL), (int)MathF.Floor(samples[i].Y / CELL));
  if (!cells.TryGetValue(k, out var l)) cells[k] = l = new List<int>();
  l.Add(i);
}
Log($"{samples.Count} samples");

GTA5Keys.LoadFromPath(game, true, null);
var rpf = new RpfManager();
rpf.Init(game, true, s => { }, e => Console.Error.WriteLine(e));
var xml = new XmlDocument(); xml.Load(Path.Combine(AppContext.BaseDirectory, "ShadersGen9Conversion.xml"));
foreach (XmlNode n in xml.SelectNodes("//Name")) JenkIndex.Ensure(n.InnerText);
foreach (XmlNode n in xml.SelectNodes("//Item[@name]")) { JenkIndex.Ensure(n.Attributes["name"].Value); if (n.Attributes["old"] != null) JenkIndex.Ensure(n.Attributes["old"].Value); }

var byExt = new Dictionary<string, Dictionary<uint, RpfFileEntry>>();
var gtxds = new List<RpfFileEntry>();
foreach (var r in rpf.AllRpfs)
{
  if (!Vanilla(r.Path)) continue;
  foreach (var e in r.AllEntries)
    if (e is RpfFileEntry fe)
    {
      if (e.NameLower == "gtxd.ymt" || e.NameLower == "gtxd.meta") gtxds.Add(fe);
      var ext = Path.GetExtension(e.NameLower);
      if (ext is ".ymap" or ".ytyp" or ".ydr" or ".ydd" or ".yft" or ".ytd")
      {
        if (!byExt.TryGetValue(ext, out var d)) byExt[ext] = d = new();
        d[JenkHash.GenHash(Path.GetFileNameWithoutExtension(e.NameLower))] = fe;
      }
    }
}
var parentTxd = new Dictionary<uint, uint>();
foreach (var fe in gtxds)
  try { var g = rpf.GetFile<GtxdFile>(fe); if (g?.TxdRelationships != null) foreach (var kv in g.TxdRelationships) parentTxd.TryAdd(JenkHash.GenHash(kv.Key.ToLowerInvariant()), JenkHash.GenHash(kv.Value.ToLowerInvariant())); } catch { }
var arch = new Dictionary<uint, Archetype>();
foreach (var fe in byExt[".ytyp"].Values)
  try { var y = rpf.GetFile<YtypFile>(fe); if (y?.AllArchetypes != null) foreach (var a in y.AllArchetypes) arch[a.Hash] = a; } catch { }
Log($"{arch.Count} archetypes");

var ytdCache = new Dictionary<uint, YtdFile>();
YtdFile Ytd(uint h) { if (!ytdCache.TryGetValue(h, out var y)) { y = byExt[".ytd"].TryGetValue(h, out var e) ? rpf.GetFile<YtdFile>(e) : null; ytdCache[h] = y; } return y; }
Texture FindTex(string name, DrawableBase dr, uint td)
{
  var th = JenkHash.GenHash(name.ToLowerInvariant());
  var t = dr?.ShaderGroup?.TextureDictionary?.Lookup(th);
  for (int i = 0; t == null && td != 0 && i < 10; i++) { t = Ytd(td)?.TextureDict?.Lookup(th); if (t == null && !parentTxd.TryGetValue(td, out td)) break; }
  return t;
}
var lookups = new Dictionary<string, Lookup>();
Lookup GetLookup(TextureBase tb, DrawableBase dr, uint td)
{
  var key = tb.Name.ToLowerInvariant();
  if (lookups.TryGetValue(key, out var lk)) return lk;
  lookups[key] = null;
  var tx = tb is Texture t0 && t0.Data?.FullData != null ? t0 : FindTex(tb.Name, dr, td);
  if (tx == null) { Log($"lookup texture {key} not found"); return null; }
  byte[] px;
  try { px = CodeWalker.Utils.DDSIO.GetPixels(tx, 0); } catch (Exception ex) { Log($"lookup texture {key}: {ex.Message}"); return null; }
  if (px == null || px.Length < tx.Width * tx.Height * 4) return null;
  return lookups[key] = new Lookup { Name = key, W = tx.Width, H = tx.Height, Px = px };
}

var seen = new HashSet<string>();
var shadersLogged = new HashSet<string>();
int nYmaps = 0, nGeo = 0;
foreach (var (h, fe) in byExt[".ymap"])
{
  YmapFile ym;
  try { ym = rpf.GetFile<YmapFile>(fe); } catch { continue; }
  if (++nYmaps % 2000 == 0) Log($"{nYmaps} ymaps, {nGeo} geometries over the samples");
  if (ym?.AllEntities == null) continue;
  var md = ym._CMapData;
  if (md.entitiesExtentsMax.X < ex0 || md.entitiesExtentsMin.X > ex1 || md.entitiesExtentsMax.Y < ey0 || md.entitiesExtentsMin.Y > ey1) continue;
  foreach (var en in ym.AllEntities)
  {
    var lod = en._CEntityDef.lodLevel;
    if (lod != rage__eLodType.LODTYPES_DEPTH_HD && lod != rage__eLodType.LODTYPES_DEPTH_ORPHANHD) continue;
    var an = en._CEntityDef.archetypeName;
    if (!arch.TryGetValue(an.Hash, out var a)) continue;
    var p = en.Position;
    if (p.X + a.BSRadius < ex0 || p.X - a.BSRadius > ex1 || p.Y + a.BSRadius < ey0 || p.Y - a.BSRadius > ey1) continue;
    if (!seen.Add(string.Format(inv, "{0}@{1:F2},{2:F2},{3:F2}", an, p.X, p.Y, p.Z))) continue; // SP and MP copies of the same placement
    var ans = an.ToString();
    if (ans.StartsWith("prop_")) continue;
    DrawableBase dr = null;
    try
    {
      if (byExt[".ydr"].TryGetValue(an.Hash, out var ydre)) dr = rpf.GetFile<YdrFile>(ydre)?.Drawable;
      else if (a.DrawableDict.Hash != 0 && byExt[".ydd"].TryGetValue(a.DrawableDict.Hash, out var ydde)) { Drawable d2 = null; rpf.GetFile<YddFile>(ydde)?.Dict?.TryGetValue(an.Hash, out d2); dr = d2; }
      else if (byExt[".yft"].TryGetValue(an.Hash, out var yfte)) dr = rpf.GetFile<YftFile>(yfte)?.Fragment?.Drawable;
    }
    catch { continue; }
    if (dr?.DrawableModels?.High == null) continue;
    foreach (var m in dr.DrawableModels.High)
      foreach (var g in m.Geometries)
      {
        var vd = g.VertexData; var ib = g.IndexBuffer?.Indices;
        if (vd?.Info == null || ib == null) continue;
        var sh = g.Shader; var pl = sh?.ParametersList;
        var shName = (sh?.Name.ToString() ?? "").ToLowerInvariant();
        bool terrain = shName.StartsWith("terrain_cb");
        var layers = new string[4]; string diff = null; TextureBase lookupTb = null;
        if (pl?.Parameters != null)
          for (int i = 0; i < pl.Parameters.Length; i++)
            if (pl.Parameters[i].Data is TextureBase tb)
            {
              var pn = pl.Hashes[i].ToString().ToLowerInvariant();
              if (pn is "diffusesampler" or "diffusetex") diff ??= tb.Name.ToLowerInvariant();
              else if (pn.StartsWith("lookup")) lookupTb = tb;
              else if ((pn.StartsWith("texturesampler_layer") || pn.StartsWith("diffusetexture_layer")) && pn[^1] is >= '0' and <= '3') layers[pn[^1] - '0'] = tb.Name.ToLowerInvariant();
            }
        if (terrain && shadersLogged.Add(shName))
          Log($"shader {shName}: vertex flags 0x{vd.Info.Flags:X} params " + string.Join(" ", Enumerable.Range(0, pl.Parameters.Length).Where(i => pl.Parameters[i].Data is TextureBase).Select(i => pl.Hashes[i].ToString())));
        uint fl = vd.Info.Flags;
        bool hasC0 = (fl & (1u << 4)) != 0, hasC1 = (fl & (1u << 5)) != 0, hasT0 = (fl & (1u << 6)) != 0, hasT1 = (fl & (1u << 7)) != 0;
        Vector2 Uv(int v, int c) => vd.Info.GetComponentType(c) == VertexComponentType.Half2 ? (Vector2)vd.GetHalf2(v, c) : vd.GetVector2(v, c);
        if (!terrain && samples.Count == 0) continue;
        Lookup lk = null; bool lkTried = false;
        // TerrainPS.hlsl: vc1 = m*(1 - vc0.a) + vc1*vc0.a, with m the lookup texture, or vc1 when there is none
        float[] Blend(int i0, int i1, int i2, float l0, float l1, float l2, Hit hit)
        {
          Vector4 Col(int c) => l0 * vd.GetColour(i0, c).ToVector4() + l1 * vd.GetColour(i1, c).ToVector4() + l2 * vd.GetColour(i2, c).ToVector4();
          Vector4? vc0 = hasC0 ? Col(4) : null, vc1 = hasC1 ? Col(5) : null, mask = null;
          if (lookupTb != null && !lkTried) { lkTried = true; lk = GetLookup(lookupTb, dr, a.TextureDict.Hash); }
          if (lk != null && hasT0)
          {
            int tc = hasT1 ? 7 : 6;
            var uv = l0 * Uv(i0, tc) + l1 * Uv(i1, tc) + l2 * Uv(i2, tc);
            mask = lk.Sample(uv.X, uv.Y);
          }
          Vector4 blend;
          if (vc1 == null) blend = mask ?? new Vector4(0, 0, 0, 1);
          else if (mask != null && vc0 != null) blend = mask.Value * (1 - vc0.Value.W) + vc1.Value * vc0.Value.W;
          else blend = vc1.Value;
          if (hit != null) { hit.Vc0 = vc0; hit.Vc1 = vc1; hit.Mask = mask; hit.Lookup = lookupTb == null ? null : lk != null ? lk.Name : lookupTb.Name.ToLowerInvariant() + " (not read)"; }
          float bb = blend.Z, gg = blend.Y;
          return new[] { (1 - bb) * (1 - gg), bb * (1 - gg), (1 - bb) * gg, bb * gg };
        }
        var W = new Vector3[vd.VertexCount];
        for (int v = 0; v < W.Length; v++) W[v] = p + en.Orientation.Multiply(vd.GetVector3(v, 0) * en.Scale);
        int ni = (int)g.IndicesCount;
        bool sv = survey != null && terrain, used = false;
        double area = 0; var share = new double[4];
        float gx0 = float.MaxValue, gx1 = float.MinValue, gy0 = float.MaxValue, gy1 = float.MinValue;
        for (int k = 0; k + 2 < ni; k += 3)
        {
          int i0 = ib[k], i1 = ib[k + 1], i2 = ib[k + 2];
          Vector3 A = W[i0], B = W[i1], C = W[i2];
          float minx = MathF.Min(A.X, MathF.Min(B.X, C.X)), maxx = MathF.Max(A.X, MathF.Max(B.X, C.X)), miny = MathF.Min(A.Y, MathF.Min(B.Y, C.Y)), maxy = MathF.Max(A.Y, MathF.Max(B.Y, C.Y));
          if (maxx - minx > 500 || maxy - miny > 500) continue;
          float d = (B.Y - C.Y) * (A.X - C.X) + (C.X - B.X) * (A.Y - C.Y);
          if (sv)
          {
            float ta = MathF.Abs(d) / 2; var tw = Blend(i0, i1, i2, 1 / 3f, 1 / 3f, 1 / 3f, null);
            area += ta; for (int j = 0; j < 4; j++) share[j] += ta * tw[j];
            gx0 = MathF.Min(gx0, minx); gx1 = MathF.Max(gx1, maxx); gy0 = MathF.Min(gy0, miny); gy1 = MathF.Max(gy1, maxy);
          }
          if (maxx < bx0 || minx > bx1 || maxy < by0 || miny > by1 || MathF.Abs(d) < 1e-9f) continue;
          for (int cx = (int)MathF.Floor(minx / CELL); cx <= (int)MathF.Floor(maxx / CELL); cx++)
            for (int cy = (int)MathF.Floor(miny / CELL); cy <= (int)MathF.Floor(maxy / CELL); cy++)
            {
              if (!cells.TryGetValue(CellKey(cx, cy), out var list)) continue;
              foreach (var si in list)
              {
                var s = samples[si];
                float l0 = ((B.Y - C.Y) * (s.X - C.X) + (C.X - B.X) * (s.Y - C.Y)) / d;
                float l1 = ((C.Y - A.Y) * (s.X - C.X) + (A.X - C.X) * (s.Y - C.Y)) / d;
                float l2 = 1 - l0 - l1;
                if (l0 < -1e-5f || l1 < -1e-5f || l2 < -1e-5f) continue;
                var hit = new Hit { Z = l0 * A.Z + l1 * B.Z + l2 * C.Z, Ent = ans, Shader = shName, Kind = terrain ? "terrain" : shName.Contains("decal") ? "decal" : "other", Diffuse = diff };
                if (terrain) { hit.Layers = layers; hit.Weights = Blend(i0, i1, i2, l0, l1, l2, hit); }
                else if (hasC0) hit.Vc0 = l0 * vd.GetColour(i0, 4).ToVector4() + l1 * vd.GetColour(i1, 4).ToVector4() + l2 * vd.GetColour(i2, 4).ToVector4();
                s.Hits.Add(hit); used = true;
              }
            }
        }
        if (used) nGeo++;
        if (sv && area > 0)
          survey.WriteLine(string.Format(inv, "{0}\t{1}\t{2:F1}\t{3:F1}\t{4:F1}\t{5:F1}\t{6:F1}\t{7}\t{8}\t", ans, shName, gx0, gy0, gx1, gy1, area, ni / 3, lookupTb?.Name.ToLowerInvariant() ?? "")
            + string.Join(" ", Enumerable.Range(0, 4).Select(j => string.Format(inv, "{0}={1:F3}", layers[j] ?? "?", share[j] / area))));
      }
  }
}
survey?.Flush();
Log($"{nYmaps} ymaps read, {nGeo} geometries near the samples");

using (var w = new StreamWriter(outPath))
  foreach (var s in samples)
  {
    s.Hits.Sort((x, y) => y.Z.CompareTo(x.Z));
    var sb = new StringBuilder();
    sb.Append(string.Format(inv, "{{\"label\":\"{0}\",\"s\":{1},\"x\":{2:F2},\"y\":{3:F2},\"hits\":[", s.Label, float.IsNaN(s.S) ? "null" : s.S.ToString("F2", inv), s.X, s.Y));
    for (int i = 0; i < s.Hits.Count; i++)
    {
      var hh = s.Hits[i];
      sb.Append(string.Format(inv, "{0}{{\"z\":{1:F2},\"ent\":\"{2}\",\"shader\":\"{3}\",\"kind\":\"{4}\"", i > 0 ? "," : "", hh.Z, hh.Ent, hh.Shader, hh.Kind));
      if (hh.Diffuse != null) sb.Append($",\"diffuse\":\"{hh.Diffuse}\"");
      if (hh.Layers != null)
        sb.Append(",\"layers\":[" + string.Join(",", Enumerable.Range(0, 4).Select(j => string.Format(inv, "{{\"tex\":{0},\"w\":{1:F3}}}", hh.Layers[j] == null ? "null" : $"\"{hh.Layers[j]}\"", hh.Weights[j]))) + "]");
      if (hh.Lookup != null) sb.Append($",\"lookup\":\"{hh.Lookup}\"");
      string V4(Vector4 v) => string.Format(inv, "[{0:F3},{1:F3},{2:F3},{3:F3}]", v.X, v.Y, v.Z, v.W);
      if (hh.Mask != null) sb.Append(",\"mask\":" + V4(hh.Mask.Value));
      if (hh.Vc0 != null) sb.Append(",\"vc0\":" + V4(hh.Vc0.Value));
      if (hh.Vc1 != null) sb.Append(",\"vc1\":" + V4(hh.Vc1.Value));
      sb.Append('}');
    }
    sb.Append("]}");
    w.WriteLine(sb.ToString());
  }

if (!quiet)
  foreach (var grp in samples.Where(s => !float.IsNaN(s.S)).GroupBy(s => s.Label))
  {
    Console.WriteLine($"== {grp.Key}  (s = metres to the right of the path; top terrain mesh, its strongest layer and weight, the other layers, then what lies on it)");
    foreach (var s in grp)
    {
      var t = s.Hits.FirstOrDefault(q => q.Kind == "terrain");
      var line = string.Format(inv, "{0,6:F1} ", s.S);
      if (t == null) line += s.Hits.Count == 0 ? "(nothing)" : $"z={s.Hits[0].Z.ToString("F1", inv)} {s.Hits[0].Shader} {s.Hits[0].Diffuse}";
      else
      {
        // the same texture can sit in more than one layer slot: add those up
        var ord = Enumerable.Range(0, 4).GroupBy(j => t.Layers[j] ?? "?").Select(q => (tex: q.Key, w: q.Sum(j => t.Weights[j]))).OrderByDescending(q => q.w).ToArray();
        line += string.Format(inv, "z={0,7:F1} {1,-24} {2,-34} {3:F2}", t.Z, t.Ent, ord[0].tex, ord[0].w);
        line += "  | " + string.Join(" ", ord.Skip(1).Where(q => q.w >= 0.05f).Select(q => string.Format(inv, "{0} {1:F2}", q.tex, q.w)));
        var over = s.Hits.Where(q => q != t && q.Z > t.Z - 0.5f && q.Z < t.Z + 1.5f).Select(q => q.Kind == "terrain" ? $"terrain:{q.Ent}" : $"{q.Kind}:{q.Diffuse ?? q.Shader}").Distinct().ToArray();
        if (over.Length > 0) line += "  || " + string.Join(" ", over);
      }
      Console.WriteLine(line);
    }
  }
Log("done");
return 0;

// only the vanilla archives: skip mod folders that sit beside them
static bool Vanilla(string p) { var l = p.ToLowerInvariant(); return l.StartsWith("x64") || l.StartsWith("update\\") || l.StartsWith("common"); }

public class Sample
{
  public string Label; public float S, X, Y;
  public List<Hit> Hits = new();
}

public class Hit
{
  public float Z; public string Ent, Shader, Kind, Diffuse, Lookup;
  public string[] Layers; public float[] Weights;
  public Vector4? Vc0, Vc1, Mask;
}

// a lookup (blend) texture's pixels, BGRA as DDSIO gives them, sampled bilinearly with wrap; returned as r, g, b, a in 0..1
public class Lookup
{
  public string Name; public int W, H; public byte[] Px;
  public Vector4 Sample(float u, float v)
  {
    float fx = u * W - 0.5f, fy = v * H - 0.5f;
    int x0 = (int)MathF.Floor(fx), y0 = (int)MathF.Floor(fy);
    float tx = fx - x0, ty = fy - y0;
    Vector4 P(int x, int y)
    {
      x = ((x % W) + W) % W; y = ((y % H) + H) % H;
      int o = 4 * (y * W + x);
      return new Vector4(Px[o + 2], Px[o + 1], Px[o], Px[o + 3]) / 255f;
    }
    return (P(x0, y0) * (1 - tx) + P(x0 + 1, y0) * tx) * (1 - ty) + (P(x0, y0 + 1) * (1 - tx) + P(x0 + 1, y0 + 1) * tx) * ty;
  }
}
