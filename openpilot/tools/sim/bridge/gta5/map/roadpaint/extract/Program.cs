// extract <game> <outdir> <x> <y> <radius> [texregex]: writes world-space vertices (x,y,z,u,v,alpha) and triangles of
// geometries whose diffuse texture matches texregex to <outdir>/geoms.jsonl, and those textures as raw RGBA to
// <outdir>/tex/. Only reads the game's files. GF_STATS=1 only counts what a run would write (e.g. for the whole map).
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Xml;
using CodeWalker.GameFiles;
using SharpDX;
using CodeWalker;

var game = args[0]; var outDir = args[1];
float qx = float.Parse(args[2]), qy = float.Parse(args[3]), qr = float.Parse(args[4]);
var texRe = new Regex(args.Length > 5 ? args[5] : "roadmark|marking|crossing|kerb|road_edge|arrow|line|junction|hatch|stop|park|paint", RegexOptions.IgnoreCase);
Directory.CreateDirectory(Path.Combine(outDir, "tex"));
var inv = System.Globalization.CultureInfo.InvariantCulture;
var sw = System.Diagnostics.Stopwatch.StartNew();
GTA5Keys.LoadFromPath(game, true, null);
var rpf = new RpfManager();
rpf.Init(game, true, s => { }, e => Console.Error.WriteLine(e));
var x = new XmlDocument(); x.Load(Path.Combine(AppContext.BaseDirectory, "ShadersGen9Conversion.xml"));
foreach (XmlNode n in x.SelectNodes("//Name")) JenkIndex.Ensure(n.InnerText);
foreach (XmlNode n in x.SelectNodes("//Item[@name]")) { JenkIndex.Ensure(n.Attributes["name"].Value); if (n.Attributes["old"] != null) JenkIndex.Ensure(n.Attributes["old"].Value); }

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
Console.Error.WriteLine($"{arch.Count} archetypes, {parentTxd.Count} txd parents {sw.Elapsed}");

var ytdCache = new Dictionary<uint, YtdFile>();
YtdFile Ytd(uint h) { if (!ytdCache.TryGetValue(h, out var y)) { y = byExt[".ytd"].TryGetValue(h, out var e) ? rpf.GetFile<YtdFile>(e) : null; ytdCache[h] = y; } return y; }
Texture FindTex(string name, DrawableBase dr, uint td)
{
  var th = JenkHash.GenHash(name.ToLowerInvariant());
  var t = dr?.ShaderGroup?.TextureDictionary?.Lookup(th);
  for (int i = 0; t == null && td != 0 && i < 10; i++) { t = Ytd(td)?.TextureDict?.Lookup(th); if (t == null && !parentTxd.TryGetValue(td, out td)) break; }
  return t;
}

var lo = new Vector2(qx - qr, qy - qr); var hi = new Vector2(qx + qr, qy + qr);
var seen = new HashSet<string>();
var texSaved = new HashSet<string>();
var texCounts = new Dictionary<string, int>();
var stats = Environment.GetEnvironmentVariable("GF_STATS") == "1";
using var w = stats ? StreamWriter.Null : new StreamWriter(Path.Combine(outDir, "geoms.jsonl"));
long nBytes = 0, nGeoms = 0, nTris = 0, nEnts = 0;
foreach (var (h, fe) in byExt[".ymap"])
{
  YmapFile ym;
  try { ym = rpf.GetFile<YmapFile>(fe); } catch { continue; }
  if (ym?.AllEntities == null) continue;
  var md = ym._CMapData;
  if (md.entitiesExtentsMax.X < lo.X || md.entitiesExtentsMin.X > hi.X || md.entitiesExtentsMax.Y < lo.Y || md.entitiesExtentsMin.Y > hi.Y) continue;
  foreach (var en in ym.AllEntities)
  {
    var lod = en._CEntityDef.lodLevel;
    if (lod != rage__eLodType.LODTYPES_DEPTH_HD && lod != rage__eLodType.LODTYPES_DEPTH_ORPHANHD) continue;
    var an = en._CEntityDef.archetypeName;
    if (!arch.TryGetValue(an.Hash, out var a)) continue;
    var p = en.Position;
    if (p.X + a.BSRadius < lo.X || p.X - a.BSRadius > hi.X || p.Y + a.BSRadius < lo.Y || p.Y - a.BSRadius > hi.Y) continue;
    var key = string.Format(inv, "{0}@{1:F2},{2:F2},{3:F2}", an, p.X, p.Y, p.Z);
    if (!seen.Add(key)) continue; // SP and MP copies of the same placement
    if (an.ToString().StartsWith("prop_")) continue;
    nEnts++;
    DrawableBase dr = null; string src = null;
    if (byExt[".ydr"].TryGetValue(an.Hash, out var ydre)) { dr = rpf.GetFile<YdrFile>(ydre)?.Drawable; src = ydre.Path; }
    else if (a.DrawableDict.Hash != 0 && byExt[".ydd"].TryGetValue(a.DrawableDict.Hash, out var ydde)) { Drawable d2 = null; rpf.GetFile<YddFile>(ydde)?.Dict?.TryGetValue(an.Hash, out d2); dr = d2; src = ydde.Path; }
    else if (byExt[".yft"].TryGetValue(an.Hash, out var yfte)) { dr = rpf.GetFile<YftFile>(yfte)?.Fragment?.Drawable; src = yfte.Path; }
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
        texCounts[diff] = texCounts.GetValueOrDefault(diff) + 1;
        if (!texRe.IsMatch(diff)) continue;
        var vd = g.VertexData; var ib = g.IndexBuffer?.Indices;
        if (vd == null || ib == null) continue;
        var sb = new StringBuilder();
        sb.Append(string.Format(inv, "{{\"ent\":\"{0}\",\"src\":\"{1}\",\"tex\":\"{2}\",\"sh\":\"{3}\",\"pos\":[{4:F3},{5:F3},{6:F3}],\"v\":[", an, src.Replace("\\", "/"), diff, sh.Name, p.X, p.Y, p.Z));
        bool hasCol = (vd.Info.Flags & (1u << 4)) != 0;
        for (int v = 0; v < vd.VertexCount; v++)
        {
          var lp = vd.GetVector3(v, 0) * en.Scale;
          var wp = p + en.Orientation.Multiply(lp);
          var uv = vd.GetVector2(v, 6);
          int alpha = hasCol ? vd.GetColour(v, 4).A : 255;
          sb.Append(string.Format(inv, "{0}[{1:F3},{2:F3},{3:F3},{4:F5},{5:F5},{6}]", v > 0 ? "," : "", wp.X, wp.Y, wp.Z, uv.X, uv.Y, alpha));
        }
        sb.Append("],\"i\":[").Append(string.Join(",", ib.Take((int)g.IndicesCount))).Append("]}");
        nBytes += sb.Length; nGeoms++; nTris += g.IndicesCount / 3;
        w.WriteLine(sb.ToString());
        if (!stats && texSaved.Add(diff))
        {
          var t = FindTex(diff, dr, a.TextureDict.Hash);
          if (t == null) { Console.Error.WriteLine($"texture {diff} not found (td {a.TextureDict})"); continue; }
          var px = CodeWalker.Utils.DDSIO.GetPixels(t, 0);
          File.WriteAllBytes(Path.Combine(outDir, "tex", $"{diff.ToLowerInvariant()}_{t.Width}x{t.Height}_{t.Format}.rgba"), px);
        }
      }
  }
}
File.WriteAllLines(Path.Combine(outDir, "textures.txt"), texCounts.OrderByDescending(k => k.Value).Select(k => $"{k.Value}\t{k.Key}"));
Console.Error.WriteLine($"done {sw.Elapsed}: {nEnts} entities, {nGeoms} geometries, {nTris} triangles, {nBytes / 1e6:F0} MB of JSON");
return 0;

// only the vanilla archives: skip mod folders (e.g. a disabled NaturalVision install) that sit beside them
static bool Vanilla(string p) { var l = p.ToLowerInvariant(); return l.StartsWith("x64") || l.StartsWith("update\\") || l.StartsWith("common"); }
