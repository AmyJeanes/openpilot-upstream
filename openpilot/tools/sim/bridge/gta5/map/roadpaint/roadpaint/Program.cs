// roadpaint <game> <outdir> [x0 y0 x1 y1]: GTA V's road paint, straight from the game's archives (only reads them).
//
// Reads every road drawable placed in the map (HD / orphan-HD ymap entities x archetype drawables, vanilla archives
// only), decides each texture's paint in texture space (Paint.cs), and writes:
//   tiles/<tx>_<ty>.bin  64 m tiles at 5 cm: per pixel up to 3 height layers of surface class + paint code and height
//   segments.bin         exact world-space paint line pieces (texture line bands mapped through each triangle's UVs)
//   decals.jsonl         world-space geometry of the atlas decals that carry features (arrows, text, markers) and of
//                        crossings, for feature typing from the atlas cell table
//   textures.tsv         every texture used, its class and line bands
// Env: RP_THREADS (default 4), RP_ATLAS (atlas cell table JSON), RP_DUMPTEX=1 (also write the textures' pixels).
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Xml;
using CodeWalker;
using CodeWalker.GameFiles;
using SharpDX;

const float TILE = 64f, RES = 0.05f;
const int N = 1280, MAXL = 3;
const int NONE = 0, ROAD = 1, KERB = 2, WALK = 3, GUTTER = 4;

var game = args[0]; var outDir = args[1];
float bx0 = -5000, by0 = -5000, bx1 = 6000, by1 = 9000;
if (args.Length >= 6) (bx0, by0, bx1, by1) = (float.Parse(args[2]), float.Parse(args[3]), float.Parse(args[4]), float.Parse(args[5]));
int threads = int.Parse(Environment.GetEnvironmentVariable("RP_THREADS") ?? "4");
bool dumpTex = Environment.GetEnvironmentVariable("RP_DUMPTEX") == "1";
var atlasPath = Environment.GetEnvironmentVariable("RP_ATLAS");
Directory.CreateDirectory(Path.Combine(outDir, "tiles"));
if (dumpTex) Directory.CreateDirectory(Path.Combine(outDir, "tex"));
var inv = System.Globalization.CultureInfo.InvariantCulture;
var sw = System.Diagnostics.Stopwatch.StartNew();
void Log(string s) => Console.Error.WriteLine($"[{sw.Elapsed:hh\\:mm\\:ss}] {s}");

var gutterRe = new Regex("road_edge|gutter|edgedecal", RegexOptions.IgnoreCase);
var kerbRe = new Regex("kerb|curb", RegexOptions.IgnoreCase);
var roadRe = new Regex("road|marking|blend|crossing|tarmac|asphalt|carpark|keep", RegexOptions.IgnoreCase);
var walkRe = new Regex("sidewalk|pave|concrete|ground", RegexOptions.IgnoreCase);
var crossingRe = new Regex("crossing", RegexOptions.IgnoreCase);
// decals that can be road paint; the rest (dirt, weeds, drains, posters...) have bright texels that aren't
var markingRe = new Regex("mark|line|crossing|keep|arrow|parking_objects|carpark|stop|hatch|paintdecal", RegexOptions.IgnoreCase);
int ClassOf(string t) => gutterRe.IsMatch(t) ? GUTTER : kerbRe.IsMatch(t) ? KERB : roadRe.IsMatch(t) ? ROAD : walkRe.IsMatch(t) ? WALK : NONE;

// atlas cell table: texture (lower case) -> { "lines": [{u0,u1,v0,v1,axis}], "features": [...] (used by the Python side) }
// keys are texture name prefixes; the longest matching one applies
var lineCells = new Dictionary<string, List<(float u0, float u1, float v0, float v1, char axis)>>();
var featureKeys = new HashSet<string>(); var excludeKeys = new HashSet<string>(); var atlasKeys = new List<string>();
if (atlasPath != null)
{
  using var doc = JsonDocument.Parse(File.ReadAllText(atlasPath));
  foreach (var p in doc.RootElement.EnumerateObject())
  {
    if (p.Name.StartsWith("_")) continue;
    atlasKeys.Add(p.Name);
    if (p.Value.TryGetProperty("lines", out var ls))
      lineCells[p.Name] = ls.EnumerateArray().Select(c => (c.GetProperty("u0").GetSingle(), c.GetProperty("u1").GetSingle(), c.GetProperty("v0").GetSingle(), c.GetProperty("v1").GetSingle(), c.GetProperty("axis").GetString()[0])).ToList();
    if (p.Value.TryGetProperty("features", out _)) featureKeys.Add(p.Name);
    if (p.Value.TryGetProperty("exclude", out var ex) && ex.GetBoolean()) excludeKeys.Add(p.Name);
  }
}
string AtlasKey(string name) => atlasKeys.Where(k => name.StartsWith(k)).OrderByDescending(k => k.Length).FirstOrDefault();
var featureTex = new HashSet<string>();  // texture names whose key has features

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
Log($"{arch.Count} archetypes, {parentTxd.Count} txd parents");

var ytdCache = new Dictionary<uint, YtdFile>();
YtdFile Ytd(uint h) { if (!ytdCache.TryGetValue(h, out var y)) { y = byExt[".ytd"].TryGetValue(h, out var e) ? rpf.GetFile<YtdFile>(e) : null; ytdCache[h] = y; } return y; }
Texture FindTex(string name, DrawableBase dr, uint td)
{
  var th = JenkHash.GenHash(name.ToLowerInvariant());
  var t = dr?.ShaderGroup?.TextureDictionary?.Lookup(th);
  for (int i = 0; t == null && td != 0 && i < 10; i++) { t = Ytd(td)?.TextureDict?.Lookup(th); if (t == null && !parentTxd.TryGetValue(td, out td)) break; }
  return t;
}

// textures, analysed once per (name, decal)
var texs = new Dictionary<string, Tex>();
var texList = new List<Tex>();
Tex GetTex(string name, bool decal, int cls, DrawableBase dr, uint td)
{
  var key = name.ToLowerInvariant() + (decal ? "|d" : "|s");
  if (texs.TryGetValue(key, out var t)) return t;
  texs[key] = null;
  var tx = FindTex(name, dr, td);
  if (tx == null) return null;
  byte[] px;
  try { px = CodeWalker.Utils.DDSIO.GetPixels(tx, 0); } catch { return null; }
  if (px == null || px.Length < tx.Width * tx.Height * 4) return null;
  t = new Tex { Id = texList.Count, Name = name.ToLowerInvariant(), Decal = decal, Cls = cls, W = tx.Width, H = tx.Height };
  var akey = AtlasKey(name.ToLowerInvariant());
  if (decal && akey != null && excludeKeys.Contains(akey)) return null;
  // road surfaces drawn as decals (rural roads over terrain, patches): their own surface and baked-in lines
  bool surfaceDecal = decal && !markingRe.IsMatch(name) && akey == null && cls == ROAD;
  if (decal && !surfaceDecal && !markingRe.IsMatch(name) && akey == null) return null;
  if (decal && akey != null && featureKeys.Contains(akey)) featureTex.Add(name.ToLowerInvariant());
  if (surfaceDecal)
  {
    t.SurfaceDecal = true;
    Paint.Surface(t, px);
    t.Alpha = new byte[t.W * t.H];
    for (int i = 0; i < t.W * t.H; i++) t.Alpha[i] = px[4 * i + 3];
  }
  else if (decal)
  {
    Paint.Decal(t, px);
    if (akey != null && lineCells.TryGetValue(akey, out var cells)) foreach (var c in cells) Paint.DecalBands(t, c.u0, c.u1, c.v0, c.v1, c.axis);
    if (!t.Paint.Any(p => p > 0) && !featureTex.Contains(t.Name) && !crossingRe.IsMatch(t.Name)) return null; // decals without paint don't matter
  }
  else if (cls == ROAD) Paint.Surface(t, px);
  else t.Paint = new byte[t.W * t.H];
  if (dumpTex) File.WriteAllBytes(Path.Combine(outDir, "tex", $"{t.Name}{(decal ? "" : "_s")}_{t.W}x{t.H}_bgra.rgba"), px);
  texs[key] = t;
  texList.Add(t);
  return t;
}

// geometry: world-space vertices (x, y, z, u, v, vertex alpha) and triangles, per drawable geometry
var geos = new List<Geo>();
var seen = new HashSet<string>();
using var decW = new StreamWriter(Path.Combine(outDir, "decals.jsonl"));
long nTris = 0;
int nYmaps = 0;
foreach (var (h, fe) in byExt[".ymap"])
{
  YmapFile ym;
  try { ym = rpf.GetFile<YmapFile>(fe); } catch { continue; }
  if (++nYmaps % 2000 == 0) Log($"{nYmaps} ymaps, {geos.Count} geometries, {nTris} triangles, {texList.Count} textures");
  if (ym?.AllEntities == null) continue;
  var md = ym._CMapData;
  if (md.entitiesExtentsMax.X < bx0 || md.entitiesExtentsMin.X > bx1 || md.entitiesExtentsMax.Y < by0 || md.entitiesExtentsMin.Y > by1) continue;
  foreach (var en in ym.AllEntities)
  {
    var lod = en._CEntityDef.lodLevel;
    if (lod != rage__eLodType.LODTYPES_DEPTH_HD && lod != rage__eLodType.LODTYPES_DEPTH_ORPHANHD) continue;
    var an = en._CEntityDef.archetypeName;
    if (!arch.TryGetValue(an.Hash, out var a)) continue;
    var p = en.Position;
    if (p.X + a.BSRadius < bx0 || p.X - a.BSRadius > bx1 || p.Y + a.BSRadius < by0 || p.Y - a.BSRadius > by1) continue;
    var key = string.Format(inv, "{0}@{1:F2},{2:F2},{3:F2}", an, p.X, p.Y, p.Z);
    if (!seen.Add(key)) continue; // SP and MP copies of the same placement
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
        var sh = g.Shader; var pl = sh?.ParametersList;
        string diff = null;
        if (pl?.Parameters != null)
          for (int i = 0; i < pl.Parameters.Length; i++)
            if (pl.Parameters[i].Data is TextureBase tb && (pl.Hashes[i].ToString().ToLowerInvariant() is "diffusesampler" or "diffusetex" or "texturesampler_layer0" or "diffusetexture_layer0")) { diff = tb.Name; break; }
        if (diff == null) continue;
        bool decal = sh.Name.ToString().ToLowerInvariant().Contains("decal");
        int cls = ClassOf(diff);
        if (!decal && cls == NONE) continue;
        var vd = g.VertexData; var ib = g.IndexBuffer?.Indices;
        if (vd == null || ib == null) continue;
        var t = GetTex(diff, decal, cls, dr, a.TextureDict.Hash);
        if (t == null) continue;
        bool hasCol = (vd.Info.Flags & (1u << 4)) != 0;
        var V = new float[vd.VertexCount * 6];
        for (int v = 0; v < vd.VertexCount; v++)
        {
          var wp = p + en.Orientation.Multiply(vd.GetVector3(v, 0) * en.Scale);
          var uv = vd.GetVector2(v, 6);
          V[6 * v] = wp.X; V[6 * v + 1] = wp.Y; V[6 * v + 2] = wp.Z; V[6 * v + 3] = uv.X; V[6 * v + 4] = uv.Y;
          V[6 * v + 5] = hasCol ? vd.GetColour(v, 4).A : 255;
        }
        var I = ib.Take((int)g.IndicesCount).Select(i => (int)i).ToArray();
        geos.Add(new Geo { V = V, I = I, T = t, Decal = decal, Cls = cls, Ent = ans });
        nTris += I.Length / 3;
        if (decal && (featureTex.Contains(t.Name) || crossingRe.IsMatch(t.Name)) || !decal && crossingRe.IsMatch(t.Name))
        {
          var sb = new StringBuilder();
          sb.Append(string.Format(inv, "{{\"ent\":\"{0}\",\"tex\":\"{1}\",\"decal\":{2},\"v\":[", ans, t.Name, decal ? "true" : "false"));
          for (int v = 0; v < vd.VertexCount; v++)
            sb.Append(string.Format(inv, "{0}[{1:F3},{2:F3},{3:F3},{4:F5},{5:F5},{6}]", v > 0 ? "," : "", V[6 * v], V[6 * v + 1], V[6 * v + 2], V[6 * v + 3], V[6 * v + 4], (int)V[6 * v + 5]));
          sb.Append("],\"i\":[").Append(string.Join(",", I)).Append("]}");
          decW.WriteLine(sb.ToString());
        }
      }
  }
}
decW.Flush();
Log($"loaded: {geos.Count} geometries, {nTris} triangles, {texList.Count} textures");
using (var tw = new StreamWriter(Path.Combine(outDir, "textures.tsv")))
{
  tw.WriteLine("id\tname\tdecal\tcls\tw\th\tbands");
  foreach (var t in texList)
    tw.WriteLine($"{t.Id}\t{t.Name}\t{(t.Decal ? 1 : 0)}\t{t.Cls}\t{t.W}\t{t.H}\t" + string.Join(" ", t.Bands.Select(b => string.Format(inv, "{0}{1:F3}/{2:F3}/{3}/{4:F2}", b.Axis, b.Pos, b.Width, b.Colour == 2 ? "y" : "w", b.Painted.Sum(q => q.b - q.a)))));
}

// bin triangles into tiles: opaque surfaces first, decals after, so decals land on the finished surfaces
var bins = new Dictionary<long, List<long>>();
for (int gi = 0; gi < geos.Count; gi++)
{
  var G = geos[gi];
  for (int k = 0; k + 2 < G.I.Length; k += 3)
  {
    float minx = float.MaxValue, maxx = float.MinValue, miny = float.MaxValue, maxy = float.MinValue;
    for (int q = 0; q < 3; q++) { int o = 6 * G.I[k + q]; minx = Math.Min(minx, G.V[o]); maxx = Math.Max(maxx, G.V[o]); miny = Math.Min(miny, G.V[o + 1]); maxy = Math.Max(maxy, G.V[o + 1]); }
    if (maxx - minx > 500 || maxy - miny > 500) continue; // broken triangle
    for (int tx = (int)Math.Floor(minx / TILE); tx <= (int)Math.Floor(maxx / TILE); tx++)
      for (int ty = (int)Math.Floor(miny / TILE); ty <= (int)Math.Floor(maxy / TILE); ty++)
      {
        long tk = ((long)(tx + 1000) << 16) | (long)(ty + 1000);
        if (!bins.TryGetValue(tk, out var l)) bins[tk] = l = new List<long>();
        l.Add(((long)gi << 24) | (long)(k / 3));
      }
  }
}
Log($"{bins.Count} tiles");

int done = 0;
Parallel.ForEach(bins, new ParallelOptions { MaxDegreeOfParallelism = threads }, kv =>
{
  int tx = (int)(kv.Key >> 16) - 1000, ty = (int)(kv.Key & 0xFFFF) - 1000;
  var R = new TileRaster(tx * TILE, ty * TILE + TILE, RES, N, MAXL);
  foreach (int pass in new[] { 0, 1, 2 })  // opaque surfaces, road surfaces drawn as decals, then markings on top
    foreach (var e in kv.Value)
    {
      var G = geos[(int)(e >> 24)];
      if ((!G.Decal ? 0 : G.T.SurfaceDecal ? 1 : 2) != pass) continue;
      R.Triangle(G, (int)(e & 0xFFFFFF) * 3);
    }
  R.Save(Path.Combine(outDir, "tiles", $"{tx}_{ty}.bin"), tx, ty);
  int d = System.Threading.Interlocked.Increment(ref done);
  if (d % 500 == 0) Log($"{d}/{bins.Count} tiles");
});
Log("tiles written");

// exact paint lines: each line band's iso-line through each triangle, cut to the band's painted stretches
var segs = new ConcurrentBag<byte[]>();
Parallel.For(0, geos.Count, new ParallelOptions { MaxDegreeOfParallelism = threads }, gi =>
{
  var G = geos[gi];
  if (G.T.Bands.Count == 0) return;
  using var ms = new MemoryStream();
  using var bw = new BinaryWriter(ms);
  for (int k = 0; k + 2 < G.I.Length; k += 3)
    foreach (var b in G.T.Bands)
      Segments.Emit(G, k, b, gi, bw);
  bw.Flush();
  if (ms.Length > 0) segs.Add(ms.ToArray());
});
using (var fs = File.Create(Path.Combine(outDir, "segments.bin")))
  foreach (var s in segs) fs.Write(s);
Log($"segments written ({segs.Sum(s => (long)s.Length) / Segments.RecordSize} pieces)");
using (var gw = new StreamWriter(Path.Combine(outDir, "geoms.tsv")))
{
  gw.WriteLine("geom\ttex\tent");
  for (int gi = 0; gi < geos.Count; gi++) gw.WriteLine($"{gi}\t{geos[gi].T.Id}\t{geos[gi].Ent}");
}
Log("done");
return 0;

// only the vanilla archives: skip mod folders (e.g. a disabled NaturalVision install) that sit beside them
static bool Vanilla(string p) { var l = p.ToLowerInvariant(); return l.StartsWith("x64") || l.StartsWith("update\\") || l.StartsWith("common"); }

public class Geo
{
  public float[] V;  // x, y, z, u, v, vertex alpha per vertex
  public int[] I;
  public Tex T;
  public bool Decal;
  public int Cls;
  public string Ent;
}

// One tile's top-down raster. Each pixel keeps up to MAXL surfaces more than 1.5 m apart in height (decks of bridges
// and interchanges), each its top surface's class and paint (code = class | paint << 3) and height.
public class TileRaster
{
  readonly float x0, y1, res;
  readonly int n, maxl;
  public readonly byte[] Code, Count;
  public readonly float[] Z;

  public TileRaster(float x0, float y1, float res, int n, int maxl)
  {
    (this.x0, this.y1, this.res, this.n, this.maxl) = (x0, y1, res, n, maxl);
    Code = new byte[maxl * n * n]; Z = new float[maxl * n * n]; Count = new byte[n * n];
  }

  public void Triangle(Geo G, int k)
  {
    int i0 = 6 * G.I[k], i1 = 6 * G.I[k + 1], i2 = 6 * G.I[k + 2];
    var V = G.V;
    float c0 = (V[i0] - x0) / res, r0 = (y1 - V[i0 + 1]) / res;
    float c1 = (V[i1] - x0) / res, r1 = (y1 - V[i1 + 1]) / res;
    float c2 = (V[i2] - x0) / res, r2 = (y1 - V[i2 + 1]) / res;
    float d = (r1 - r2) * (c0 - c2) + (c2 - c1) * (r0 - r2);
    if (Math.Abs(d) < 1e-9f) return;
    int cmin = Math.Max((int)Math.Floor(Math.Min(c0, Math.Min(c1, c2))), 0), cmax = Math.Min((int)Math.Ceiling(Math.Max(c0, Math.Max(c1, c2))), n - 1);
    int rmin = Math.Max((int)Math.Floor(Math.Min(r0, Math.Min(r1, r2))), 0), rmax = Math.Min((int)Math.Ceiling(Math.Max(r0, Math.Max(r1, r2))), n - 1);
    if (cmin > cmax || rmin > rmax) return;
    var T = G.T;
    int W = T.W, H = T.H, nn = n * n;
    for (int r = rmin; r <= rmax; r++)
    {
      float pr = r + 0.5f;
      for (int c = cmin; c <= cmax; c++)
      {
        float pc = c + 0.5f;
        float l0 = ((r1 - r2) * (pc - c2) + (c2 - c1) * (pr - r2)) / d;
        float l1 = ((r2 - r0) * (pc - c2) + (c0 - c2) * (pr - r2)) / d;
        float l2 = 1 - l0 - l1;
        if (l0 < -1e-5f || l1 < -1e-5f || l2 < -1e-5f) continue;
        float u = l0 * V[i0 + 3] + l1 * V[i1 + 3] + l2 * V[i2 + 3];
        float v = l0 * V[i0 + 4] + l1 * V[i1 + 4] + l2 * V[i2 + 4];
        float z = l0 * V[i0 + 2] + l1 * V[i1 + 2] + l2 * V[i2 + 2];
        int tu = (int)Math.Floor(u * W) % W; if (tu < 0) tu += W;
        int tv = (int)Math.Floor(v * H) % H; if (tv < 0) tv += H;
        byte paint = T.Paint[tv * W + tu];
        int p = r * n + c;
        int cnt = Count[p];
        if (G.Decal)
        {
          float al = T.Alpha[tv * W + tu] / 255f * ((l0 * V[i0 + 5] + l1 * V[i1 + 5] + l2 * V[i2 + 5]) / 255f);
          if (T.SurfaceDecal)
          {
            if (al < 0.5f) continue;
            byte sc = (byte)(1 | (paint << 3));
            bool on = false;
            for (int l = 0; l < cnt; l++)
            {
              float zl = Z[l * nn + p];
              if (Math.Abs(z - zl) < 0.6f) { Code[l * nn + p] = sc; Z[l * nn + p] = Math.Max(z, zl); on = true; break; }
            }
            if (!on && cnt < maxl) { Z[cnt * nn + p] = z; Code[cnt * nn + p] = sc; Count[p] = (byte)(cnt + 1); }
            continue;
          }
          if (!((al > 0.3f && paint > 0) || al > 0.85f)) continue; // dirt and wear decals over a line leave it painted
          for (int l = 0; l < cnt; l++)
          {
            float zl = Z[l * nn + p];
            if (z > zl - 0.35f && z < zl + 0.6f) { Code[l * nn + p] = (byte)((Code[l * nn + p] & 7) | (paint << 3)); break; }
          }
          continue;
        }
        byte code = (byte)(G.Cls | ((G.Cls == 1 ? paint : 0) << 3));
        bool placed = false;
        for (int l = 0; l < cnt; l++)
        {
          float zl = Z[l * nn + p];
          if (Math.Abs(z - zl) < 1.5f) { if (z > zl) { Z[l * nn + p] = z; Code[l * nn + p] = code; } placed = true; break; }
        }
        if (placed) continue;
        if (cnt < maxl) { Z[cnt * nn + p] = z; Code[cnt * nn + p] = code; Count[p] = (byte)(cnt + 1); }
        else
        {
          int lo = 0;
          for (int l = 1; l < cnt; l++) if (Z[l * nn + p] < Z[lo * nn + p]) lo = l;
          if (z > Z[lo * nn + p]) { Z[lo * nn + p] = z; Code[lo * nn + p] = code; }
        }
      }
    }
  }

  public void Save(string path, int tx, int ty)
  {
    int nl = 0;
    foreach (var c in Count) nl = Math.Max(nl, c);
    if (nl == 0) return;
    using var fs = File.Create(path);
    using var bw = new BinaryWriter(fs);
    bw.Write(0x31545052); bw.Write(tx); bw.Write(ty); bw.Write(n); bw.Write(res); bw.Write(nl);
    void Block(byte[] raw)
    {
      using var ms = new MemoryStream();
      using (var z = new ZLibStream(ms, CompressionLevel.Fastest, true)) z.Write(raw, 0, raw.Length);
      bw.Write((int)ms.Length); bw.Write(ms.ToArray());
    }
    Block(Code.AsSpan(0, nl * n * n).ToArray());
    var zb = new byte[nl * n * n * 4];
    Buffer.BlockCopy(Z, 0, zb, 0, zb.Length);
    Block(zb);
  }
}

public static class Segments
{
  // x1 y1 z1 x2 y2 z2 width (float32), colour, decal (byte), tex id (uint16), geometry id (int32)
  public const int RecordSize = 7 * 4 + 1 + 1 + 2 + 4;

  public static void Emit(Geo G, int k, Band b, int gi, BinaryWriter bw)
  {
    var V = G.V;
    int[] ix = { 6 * G.I[k], 6 * G.I[k + 1], 6 * G.I[k + 2] };
    int ac = b.Axis == 'v' ? 3 : 4, lc = b.Axis == 'v' ? 4 : 3;  // across and along coordinate offsets
    float amin = float.MaxValue, amax = float.MinValue;
    foreach (var o in ix) { amin = Math.Min(amin, V[o + ac]); amax = Math.Max(amax, V[o + ac]); }
    int kmin = b.Wraps ? (int)Math.Ceiling(amin - b.Pos) : 0, kmax = b.Wraps ? (int)Math.Floor(amax - b.Pos) : 0;
    if (kmax - kmin > 50) return;
    // world gradient of the across coordinate's inverse: metres per texture unit across, for the line's width
    var P0 = new Vector3(V[ix[0]], V[ix[0] + 1], V[ix[0] + 2]); var P1 = new Vector3(V[ix[1]], V[ix[1] + 1], V[ix[1] + 2]); var P2 = new Vector3(V[ix[2]], V[ix[2] + 1], V[ix[2] + 2]);
    float da1 = V[ix[1] + ac] - V[ix[0] + ac], db1 = V[ix[1] + lc] - V[ix[0] + lc], da2 = V[ix[2] + ac] - V[ix[0] + ac], db2 = V[ix[2] + lc] - V[ix[0] + lc];
    float det = da1 * db2 - da2 * db1;
    if (Math.Abs(det) < 1e-12f) return;
    var dPda = ((P1 - P0) * db2 - (P2 - P0) * db1) / det;
    float width = b.Width * new Vector2(dPda.X, dPda.Y).Length();
    if (width > 2f) return;  // stretched UVs: not a line here
    for (int kk = kmin; kk <= kmax; kk++)
    {
      float t = b.Pos + kk;
      if (t < amin || t > amax) continue;
      Span<Vector3> pts = stackalloc Vector3[3];
      Span<float> along = stackalloc float[3];
      int np = 0;
      for (int e = 0; e < 3 && np < 2; e++)
      {
        int a = ix[e], c = ix[(e + 1) % 3];
        float fa = V[a + ac] - t, fc = V[c + ac] - t;
        if ((fa < 0) == (fc < 0) && fa != 0) continue;
        if (fa == fc) continue;
        float s = fa / (fa - fc);
        if (s < 0 || s > 1) continue;
        var q = new Vector3(V[a] + s * (V[c] - V[a]), V[a + 1] + s * (V[c + 1] - V[a + 1]), V[a + 2] + s * (V[c + 2] - V[a + 2]));
        float al = V[a + lc] + s * (V[c + lc] - V[a + lc]);
        if (np == 1 && (q - pts[0]).LengthSquared() < 1e-8f) continue;
        pts[np] = q; along[np] = al; np++;
      }
      if (np < 2) continue;
      Vector3 A = pts[0], B = pts[1];
      float va = along[0], vb = along[1];
      if (va > vb) { (A, B) = (B, A); (va, vb) = (vb, va); }
      if (vb - va < 1e-6f) continue;
      // painted stretches [m + pa, m + pb) overlapping [va, vb]
      for (int m = (int)Math.Floor(va) - 1; m <= (int)Math.Floor(vb); m++)
        foreach (var (pa, pb) in b.Painted)
        {
          float s0 = Math.Max(va, m + pa), s1 = Math.Min(vb, m + pb);
          if (s1 - s0 < 1e-6f) continue;
          float f0 = (s0 - va) / (vb - va), f1 = (s1 - va) / (vb - va);
          var Q0 = A + (B - A) * f0; var Q1 = A + (B - A) * f1;
          bw.Write(Q0.X); bw.Write(Q0.Y); bw.Write(Q0.Z); bw.Write(Q1.X); bw.Write(Q1.Y); bw.Write(Q1.Z); bw.Write(width);
          bw.Write(b.Colour); bw.Write((byte)(G.Decal ? 1 : 0)); bw.Write((ushort)G.T.Id); bw.Write(gi);
        }
    }
  }
}
