// signprobe <game> <outdir>: every placed road sign / traffic light and every painted road word in the map, straight
// from the game's archives (only reads them). Headings are degrees anticlockwise from east (+x). Writes to <outdir>:
//   signs.jsonl     one line per placed sign: {"arch","x","y","z","heading","says","face","ymap"}. heading is the way
//                   the sign's face points, towards the drivers who read it: they travel along heading+180. "face" is
//                   the texture (and part of it) the direction was read from, or "assumed -y" where the model gave no
//                   answer. Also, where they apply: "headings" (a sign that faces several ways: a two-sided board, a
//                   four-sided post), "at":"face" (a sign that is part of a map mesh: x,y,z is its face's centre, not
//                   the mesh's origin; a prop's x,y,z is its origin, the foot of its post), "fallen" (lying down),
//                   "repeats" (other maps place the same thing there), "dlc" (only a DLC pack's map places it).
//   archetypes.tsv  one line per sign archetype: placements, what it says, the face's local direction(s) and what it
//                   was read from, size, textures, and each texture's area by the way it faces.
//   words.jsonl     one line per painted word: {"word","x","y","z","heading","tex","ent","cover","length","width",
//                   "ymap"}. heading is the way the drivers who read it travel (the way the top of the text points).
//                   cover is the share of the word's texture cell that the decal maps (a word below 0.5 is dropped:
//                   many decals only clip a cell's edge, or use the line band beside the word), length and width its
//                   size on the road. Also "alpha" (mean vertex alpha where the paint is faded), "pieces" (the word
//                   lies across several patches or road sections), "mirrored", "dlc".
// The sign's face: every sign model is read once, and per texture its triangles' area is summed by the way their vertex
// normals point. The face is the texture the Says table knows (prop_sign_road_stop_01, ...); on the sign props it
// points along the model's -y and their back (prop_sign_back_01) along +y. What a sign says comes from the same table,
// which was filled in from the texture names and by looking at the textures (SP_TEX).
// Placements: HD and orphan-HD entities of every ymap in the vanilla archives (base, update and every DLC pack; a ymap
// name found in several archives is read once, from update.rpf, else the latest patchday pack, else another pack, else
// the base archives), and the entities of placed interiors (MLOs, with all their entity sets; their "ymap" ends in
// |mlo:<interior>). LOD and SLOD entities are skipped: they are the far-away stand-ins of HD entities. An archetype
// placed at the same spot (to 10 cm) by several ymaps is written once, credited to the base map if it is in it: the
// DLC packs carry their own copies of base maps (hei_*, apa_*, lr_* ...), which are only streamed in by online content.
// Env: SP_SURVEY=<dir> also lists every small placed archetype with its textures (survey_arch.tsv), every diffuse
//      texture used on map geometry (survey_tex.tsv), which maps repeat which (survey_repeats.tsv), the UV boxes the
//      word textures' decals use (survey_word_uv.tsv) and the dropped words (survey_partial_words.jsonl): for finding
//      sign models, word textures and word cells that the tables here miss;
//      SP_TEX=<dir> writes the sign archetypes' and word decals' textures as PNG;
//      SP_MODEL=<arch,arch> writes those sign archetypes' triangles (model-space position, normal, UV, texture, bone)
//      to <arch>.tris.jsonl beside the PNGs, for checking a face direction by hand.
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Xml;
using CodeWalker;
using CodeWalker.GameFiles;
using SharpDX;

var inv = System.Globalization.CultureInfo.InvariantCulture;
var game = args[0]; var outDir = args[1];
var surveyDir = Environment.GetEnvironmentVariable("SP_SURVEY");
var texDir = Environment.GetEnvironmentVariable("SP_TEX");
var dumpModels = (Environment.GetEnvironmentVariable("SP_MODEL") ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries).ToHashSet();
Directory.CreateDirectory(outDir);
if (surveyDir != null) Directory.CreateDirectory(surveyDir);
if (texDir != null) Directory.CreateDirectory(texDir);
var sw = System.Diagnostics.Stopwatch.StartNew();
void Log(string s) => Console.Error.WriteLine($"[{sw.Elapsed:hh\\:mm\\:ss}] {s}");

// sign models by name, and by texture for models whose name doesn't say
var signNameRe = new Regex("^prop_(snow_)?sign_|^prop_traffic|road_?sign|stop_?sign|street_?sign|sign_road|traffic_?light", RegexOptions.IgnoreCase);
var signTexRe = new Regex("prop_sign_road|road_?sign|stop_?sign|yield|give_?way|speed_?limit|traffic_?light", RegexOptions.IgnoreCase);
// textures that are a sign's back, post or fittings rather than its face
var notFaceRe = new Regex("back|pole|post|metal|bolt|frame|wood|concrete|rust|base|steel|wire|box|bracket", RegexOptions.IgnoreCase);

GTA5Keys.LoadFromPath(game, true, null);
var rpf = new RpfManager();
rpf.Init(game, true, s => { }, e => Console.Error.WriteLine(e));
var xml = new XmlDocument(); xml.Load(Path.Combine(AppContext.BaseDirectory, "ShadersGen9Conversion.xml"));
foreach (XmlNode n in xml.SelectNodes("//Name")) JenkIndex.Ensure(n.InnerText);
foreach (XmlNode n in xml.SelectNodes("//Item[@name]")) { JenkIndex.Ensure(n.Attributes["name"].Value); if (n.Attributes["old"] != null) JenkIndex.Ensure(n.Attributes["old"].Value); }

var byExt = new Dictionary<string, Dictionary<uint, RpfFileEntry>>();
var gtxds = new List<RpfFileEntry>();
int ymapCopies = 0;
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
        var h = JenkHash.GenHash(Path.GetFileNameWithoutExtension(e.NameLower));
        if (ext == ".ymap" && d.ContainsKey(h)) ymapCopies++;
        if (!d.TryGetValue(h, out var old) || Priority(fe.Path) >= Priority(old.Path)) d[h] = fe;
      }
    }
}
var parentTxd = new Dictionary<uint, uint>();
foreach (var fe in gtxds)
  try { var g = rpf.GetFile<GtxdFile>(fe); if (g?.TxdRelationships != null) foreach (var kv in g.TxdRelationships) parentTxd.TryAdd(JenkHash.GenHash(kv.Key.ToLowerInvariant()), JenkHash.GenHash(kv.Value.ToLowerInvariant())); } catch { }
var arch = new Dictionary<uint, Archetype>();
foreach (var fe in byExt[".ytyp"].Values.OrderBy(f => Priority(f.Path)))
  try { var y = rpf.GetFile<YtypFile>(fe); if (y?.AllArchetypes != null) foreach (var a in y.AllArchetypes) arch[a.Hash] = a; } catch { }
Log($"{arch.Count} archetypes, {byExt[".ymap"].Count} ymaps ({ymapCopies} more copies of the same names in other archives)");

var ytdCache = new Dictionary<uint, YtdFile>();
YtdFile Ytd(uint h) { if (!ytdCache.TryGetValue(h, out var y)) { y = byExt[".ytd"].TryGetValue(h, out var e) ? rpf.GetFile<YtdFile>(e) : null; ytdCache[h] = y; } return y; }
Texture FindTex(string name, DrawableBase dr, uint td)
{
  var th = JenkHash.GenHash(name.ToLowerInvariant());
  var t = dr?.ShaderGroup?.TextureDictionary?.Lookup(th);
  for (int i = 0; t == null && td != 0 && i < 10; i++) { t = Ytd(td)?.TextureDict?.Lookup(th); if (t == null && !parentTxd.TryGetValue(td, out td)) break; }
  return t;
}
DrawableBase LoadDrawable(uint h, Archetype a)
{
  try
  {
    if (byExt[".ydr"].TryGetValue(h, out var ydre)) return rpf.GetFile<YdrFile>(ydre)?.Drawable;
    if (a.DrawableDict.Hash != 0 && byExt[".ydd"].TryGetValue(a.DrawableDict.Hash, out var ydde)) { Drawable d2 = null; rpf.GetFile<YddFile>(ydde)?.Dict?.TryGetValue(h, out d2); return d2; }
    if (byExt[".yft"].TryGetValue(h, out var yfte)) return rpf.GetFile<YftFile>(yfte)?.Fragment?.Drawable;
  }
  catch { }
  return null;
}
static string Diffuse(ShaderFX sh)
{
  var pl = sh?.ParametersList;
  if (pl?.Parameters == null) return null;
  for (int i = 0; i < pl.Parameters.Length; i++)
    if (pl.Parameters[i].Data is TextureBase tb && (pl.Hashes[i].ToString().ToLowerInvariant() is "diffusesampler" or "diffusetex" or "texturesampler_layer0" or "diffusetexture_layer0")) return tb.Name?.ToLowerInvariant();
  return null;
}

// ---- placements
var places = new List<Place>();
var seen = new Dictionary<string, int>();
var repeats = new Dictionary<string, int>();
int nLod = 0, nDup = 0, nMlo = 0, nMloEnts = 0, nNoArch = 0;
void Add(MetaHash an, Vector3 p, Quaternion q, Vector3 scale, string src)
{
  if (!arch.ContainsKey(an.Hash)) { nNoArch++; return; }
  var key = string.Format(inv, "{0}@{1:F1},{2:F1},{3:F1}", an.Hash, p.X, p.Y, p.Z);
  if (seen.TryGetValue(key, out var first))
  {
    nDup++; places[first].Dups++;
    if (surveyDir != null) { var k = places[first].Src + "\t" + src; repeats[k] = repeats.GetValueOrDefault(k) + 1; }
    return;
  }
  seen[key] = places.Count;
  places.Add(new Place { Arch = an.Hash, P = p, Q = q, Scale = scale, Src = src });
}
// the base map first (with its patches), so a placement that a DLC pack's map repeats is credited to the base map
foreach (var fe in byExt[".ymap"].Values.OrderBy(f => Dlc(f.Path)).ThenBy(f => f.Path))
{
  YmapFile ym;
  try { ym = rpf.GetFile<YmapFile>(fe); } catch { continue; }
  if (ym?.AllEntities == null) continue;
  var src = fe.Path.Replace('\\', '/');
  foreach (var en in ym.AllEntities)
  {
    var lod = en._CEntityDef.lodLevel;
    if (lod != rage__eLodType.LODTYPES_DEPTH_HD && lod != rage__eLodType.LODTYPES_DEPTH_ORPHANHD) { nLod++; continue; }
    var an = en._CEntityDef.archetypeName;
    if (arch.TryGetValue(an.Hash, out var a) && a is MloArchetype mlo)
    {
      nMlo++;
      var all = (mlo.entities ?? Array.Empty<MCEntityDef>()).AsEnumerable();
      if (mlo.entitySets != null) foreach (var es in mlo.entitySets) if (es?.Entities != null) all = all.Concat(es.Entities);
      foreach (var me in all)
      {
        // an interior's entities are stored relative to it; same maths as CodeWalker's YmapEntityDef(mloParent, ent, index)
        var d = me._Data; var lq = new Quaternion(d.rotation);
        if (lq != Quaternion.Identity) lq = Quaternion.Invert(lq);
        nMloEnts++;
        Add(d.archetypeName, en.Position + en.Orientation.Multiply(d.position), Quaternion.Multiply(en.Orientation, lq), new Vector3(d.scaleXY, d.scaleXY, d.scaleZ), src + "|mlo:" + an);
      }
      continue;
    }
    Add(an, en.Position, en.Orientation, en.Scale, src);
  }
}
Log($"{places.Count} placements ({nLod} LOD entities skipped, {nDup} repeats of an archetype at the same place, {nMlo} interiors with {nMloEnts} entities, {nNoArch} without an archetype)");

if (surveyDir != null) File.WriteAllLines(Path.Combine(surveyDir, "survey_repeats.tsv"), new[] { "first\trepeat\tentities" }.Concat(repeats.OrderByDescending(k => k.Value).Select(k => $"{k.Key}\t{k.Value}")));

// ---- every placed archetype's model, once: is it a sign (and which way does it face), does it carry painted words
var byArch = places.GroupBy(p => p.Arch).ToDictionary(g => g.Key, g => g.Count());
var signs = new Dictionary<uint, SignArch>();
var wordArch = new HashSet<uint>();
var texUse = new Dictionary<string, (int n, bool decal, string shader, string ent)>();
using var surveyArch = surveyDir != null ? new StreamWriter(Path.Combine(surveyDir, "survey_arch.tsv")) : null;
surveyArch?.WriteLine("arch\tplaced\tradius\ttextures");
var pngDone = new HashSet<string>();
void SavePng(string name, DrawableBase dr, uint td)
{
  if (texDir == null || !pngDone.Add(name)) return;
  try
  {
    var t = FindTex(name, dr, td);
    if (t == null) { Log($"texture {name} not found"); return; }
    Png.Write(Path.Combine(texDir, name + ".png"), CodeWalker.Utils.DDSIO.GetPixels(t, 0), t.Width, t.Height);
  }
  catch (Exception e) { Log($"texture {name}: {e.Message}"); }
}
int nModels = 0;
foreach (var (h, count) in byArch)
{
  var a = arch[h];
  var name = a.Name ?? h.ToString();
  bool byName = signNameRe.IsMatch(name);
  bool small = a.BSRadius < 15;
  var dr = LoadDrawable(h, a);
  nModels++;
  if (nModels % 20000 == 0) Log($"{nModels}/{byArch.Count} models");
  var models = dr?.DrawableModels?.High;
  if (models == null) { if (byName) signs[h] = new SignArch { Name = name, Count = count, Face = new Vector3(0, -1, 0), Faces = { new Vector3(0, -1, 0) }, FaceTex = "", How = "no model: assumed -y" }; continue; }
  var texs = new List<string>(); var words = false;
  foreach (var m in models)
    foreach (var g in m.Geometries)
    {
      var diff = Diffuse(g.Shader);
      if (diff == null) continue;
      if (!texs.Contains(diff)) texs.Add(diff);
      if (Words.Cells(diff) != null) words = true;
      if (surveyDir != null)
      {
        var u = texUse.GetValueOrDefault(diff);
        texUse[diff] = (u.n + count, u.decal || g.Shader.Name.ToString().ToLowerInvariant().Contains("decal"), u.shader ?? g.Shader.Name.ToString(), u.ent ?? name);
      }
    }
  if (words) wordArch.Add(h);
  if (small) surveyArch?.WriteLine(string.Format(inv, "{0}\t{1}\t{2:F1}\t{3}", name, count, a.BSRadius, string.Join(" ", texs)));
  if (!(byName || small && texs.Any(t => signTexRe.IsMatch(t)))) continue;

  // which way the face points. Per texture (for the models whose one texture holds everything, per part of it that
  // tells: Says.FaceParts): the triangles' area by the compass octant their vertex normals point into.
  var per = new Dictionary<string, FaceAcc>();
  using var dump = dumpModels.Contains(name) ? new StreamWriter(Path.Combine(texDir ?? outDir, name + ".tris.jsonl")) : null;
  Vector3 blo = new(float.MaxValue), bhi = new(float.MinValue);
  var bones = dr.Skeleton?.Bones?.Items;
  foreach (var m in models)
    foreach (var g in m.Geometries)
    {
      var diff = Diffuse(g.Shader) ?? "(none)";
      var vd = g.VertexData; var ib = g.IndexBuffer?.Indices;
      if (vd == null || ib == null) continue;
      bool hasN = (vd.Info.Flags & (1u << 3)) != 0, hasUv = (vd.Info.Flags & (1u << 6)) != 0;
      // a part hung on a bone (a traffic light's heads) is stored relative to it
      var bone = m.HasSkin == 0 && bones != null && m.BoneIndex < bones.Length ? bones[m.BoneIndex] : null;
      var xf = bone?.AnimTransform ?? Matrix.Identity;
      var parts = Says.FaceParts(diff);
      FaceAcc acc = null;
      if (parts == null && !per.TryGetValue(diff, out acc)) per[diff] = acc = new FaceAcc();
      for (int i = 0; i + 2 < g.IndicesCount; i += 3)
      {
        Vector3 p0 = Vector3.TransformCoordinate(vd.GetVector3(ib[i], 0), xf), p1 = Vector3.TransformCoordinate(vd.GetVector3(ib[i + 1], 0), xf), p2 = Vector3.TransformCoordinate(vd.GetVector3(ib[i + 2], 0), xf);
        blo = Vector3.Min(blo, Vector3.Min(p0, Vector3.Min(p1, p2))); bhi = Vector3.Max(bhi, Vector3.Max(p0, Vector3.Max(p1, p2)));
        var c = Vector3.Cross(p1 - p0, p2 - p0); float ar = c.Length() / 2;
        if (ar < 1e-9f) continue;
        bool reversed = false;
        if (parts != null)
        {
          var uv = (vd.GetVector2(ib[i], 6) + vd.GetVector2(ib[i + 1], 6) + vd.GetVector2(ib[i + 2], 6)) / 3;
          uv.X -= MathF.Floor(uv.X); uv.Y -= MathF.Floor(uv.Y);
          int k = hasUv ? Array.FindIndex(parts, q => uv.X >= q.u0 && uv.X <= q.u1 && uv.Y >= q.v0 && uv.Y <= q.v1) : -1;
          var key = k < 0 ? diff + "#rest" : $"{diff}#{k + 1}:{parts[k].what}";
          if (!per.TryGetValue(key, out acc)) per[key] = acc = new FaceAcc { Rank = k < 0 ? 99 : k };
          reversed = k >= 0 && parts[k].reversed;
        }
        var nn = hasN ? Vector3.TransformNormal(vd.GetVector3(ib[i], 3) + vd.GetVector3(ib[i + 1], 3) + vd.GetVector3(ib[i + 2], 3), xf) : c;
        if (nn.Length() < 1e-6f) continue;
        nn.Normalize();
        if (dump != null)
        {
          Vector2 t0 = vd.GetVector2(ib[i], 6), t1 = vd.GetVector2(ib[i + 1], 6), t2 = vd.GetVector2(ib[i + 2], 6);
          dump.WriteLine(string.Format(inv, "{{\"tex\":\"{0}\",\"bone\":\"{19}\",\"p\":[{1:F4},{2:F4},{3:F4},{4:F4},{5:F4},{6:F4},{7:F4},{8:F4},{9:F4}],\"n\":[{10:F3},{11:F3},{12:F3}],\"uv\":[{13:F4},{14:F4},{15:F4},{16:F4},{17:F4},{18:F4}]}}",
            diff, p0.X, p0.Y, p0.Z, p1.X, p1.Y, p1.Z, p2.X, p2.Y, p2.Z, nn.X, nn.Y, nn.Z, t0.X, t0.Y, t1.X, t1.Y, t2.X, t2.Y, bone?.Name));
        }
        if (reversed) nn = -nn;
        acc.Area += ar;
        if (MathF.Abs(nn.Z) > 0.7f) continue;  // faces up or down
        int oct = ((int)MathF.Round(MathF.Atan2(nn.Y, nn.X) / (MathF.PI / 4)) + 8) % 8;
        acc.Oct[oct] += ar; acc.N[oct] += nn * ar; acc.C[oct] += (p0 + p1 + p2) / 3 * ar;
      }
    }
  var sa = new SignArch { Name = name, Count = count, Lo = blo, Hi = bhi, Textures = texs };
  // the face: the texture this tool knows as a sign face (else any that isn't a back or a post) with the most upright area
  string best = null; double bestArea = 0; bool bestKnown = false; int bestRank = 0;
  foreach (var (t, v) in per)
  {
    if (v.Area <= 0) continue;
    sa.Detail.Add(string.Format(inv, "{0}:area={1:F2},faces={2}", t, v.Area, v.Describe(inv)));
    bool known = Says.OfTexture(t.Split('#')[0]) != null;
    if (v.Rank == 99 || !known && (bestKnown || notFaceRe.IsMatch(t) && per.Count > 1)) continue;
    double up = v.Oct.Sum();
    if (up < 0.3 * v.Area || up < 0.05) continue;
    if (known && !bestKnown || v.Rank < bestRank || v.Rank == bestRank && up > bestArea) { bestArea = up; best = t; bestKnown = known; bestRank = v.Rank; }
  }
  if (best != null)
  {
    var v = per[best];
    var sum = v.N.Aggregate((a, b) => a + b); sum.Z = 0;
    if (sum.Length() > 0.6 * v.Oct.Sum())
    {
      // one-sided (flat, or curved like a signal head's back): the mean normal
      sa.FaceCentre = v.C.Aggregate((a, b) => a + b) / (float)v.Oct.Sum(); sum.Normalize(); sa.Faces.Add(sum); sa.How = "model";
    }
    else
    {
      // faces several ways (two-sided boards, four-sided posts): each way, the biggest first
      var order = Enumerable.Range(0, 8).Where(o => v.Oct[o] > 0.2 * v.Oct.Max()).OrderByDescending(o => v.Oct[o]).ThenBy(o => o).ToArray();
      foreach (var o in order) { var n = v.N[o]; n.Z = 0; n.Normalize(); sa.Faces.Add(n); }
      sa.FaceCentre = v.C[order[0]] / (float)v.Oct[order[0]]; sa.How = order.Length == 1 ? "model" : "model, " + order.Length + " ways";
    }
    sa.Face = sa.Faces[0]; sa.FaceTex = best;
  }
  else { sa.Face = new Vector3(0, -1, 0); sa.Faces.Add(sa.Face); sa.FaceTex = ""; sa.How = "assumed -y"; }
  // what it says: every known face texture on it, the face's own first, then by size
  sa.Says = Says.OfArchetype(name) ?? string.Join(" + ", per.Where(k => Says.OfTexture(k.Key.Split('#')[0]) != null).OrderByDescending(k => k.Key == best).ThenByDescending(k => k.Value.Area).Select(k => Says.OfTexture(k.Key.Split('#')[0])).Distinct());
  signs[h] = sa;
  foreach (var t in texs) SavePng(t, dr, a.TextureDict.Hash);
}
Log($"{nModels} models read: {signs.Count} sign archetypes, {wordArch.Count} archetypes with word decals");
if (surveyDir != null)
  File.WriteAllLines(Path.Combine(surveyDir, "survey_tex.tsv"), new[] { "tex\tplaced\tdecal\tshader\texample" }.Concat(texUse.OrderByDescending(k => k.Value.n).Select(k => $"{k.Key}\t{k.Value.n}\t{(k.Value.decal ? 1 : 0)}\t{k.Value.shader}\t{k.Value.ent}")));

// ---- signs
int nSigns = 0, nSignsDlc = 0;
using (var w = new StreamWriter(Path.Combine(outDir, "signs.jsonl")))
  foreach (var pl in places)
  {
    if (!signs.TryGetValue(pl.Arch, out var sa)) continue;
    double Heading(Vector3 local) { var f = pl.Q.Multiply(local); return Math.Atan2(f.Y, f.X) * 180 / Math.PI; }
    var up = pl.Q.Multiply(new Vector3(0, 0, 1));
    // a map mesh's origin can be anywhere: the sign is where its face is. A prop's origin is the foot of its post.
    bool atFace = sa.How.StartsWith("model") && !sa.Name.StartsWith("prop_");
    var p = atFace ? pl.P + pl.Q.Multiply(sa.FaceCentre * pl.Scale) : pl.P;
    var sb = new StringBuilder();
    sb.Append(string.Format(inv, "{{\"arch\":\"{0}\",\"x\":{1:F3},\"y\":{2:F3},\"z\":{3:F3},\"heading\":{4:F1},\"says\":\"{5}\",\"face\":\"{6}\"", sa.Name, p.X, p.Y, p.Z, Heading(sa.Face), sa.Says, sa.How.StartsWith("model") ? sa.FaceTex : sa.How));
    if (sa.Faces.Count > 1) sb.Append(",\"headings\":[").Append(string.Join(",", sa.Faces.Select(f => Heading(f).ToString("F1", inv)))).Append(']');
    if (atFace) sb.Append(",\"at\":\"face\"");
    if (up.Z < 0.7f) sb.Append(",\"fallen\":true");  // knocked over or lying down: no heading worth the name
    if (pl.Dups > 0) sb.Append($",\"repeats\":{pl.Dups}");
    if (Dlc(pl.Src)) { sb.Append(",\"dlc\":true"); nSignsDlc++; }
    sb.Append($",\"ymap\":\"{pl.Src}\"}}");
    w.WriteLine(sb.ToString()); nSigns++;
  }
using (var w = new StreamWriter(Path.Combine(outDir, "archetypes.tsv")))
{
  w.WriteLine("arch\tplaced\tsays\tface_local\tface_from\tsize_xyz\ttextures\tper_texture_faces");
  foreach (var sa in signs.Values.OrderBy(s => s.Name))
    w.WriteLine(string.Format(inv, "{0}\t{1}\t{2}\t{3}\t{4}\t{5:F2}x{6:F2}x{7:F2}\t{8}\t{9}", sa.Name, sa.Count, sa.Says, string.Join(" ", sa.Faces.Select(f => string.Format(inv, "({0:F2},{1:F2})", f.X, f.Y))),
      sa.How.StartsWith("model") ? sa.How + ": " + sa.FaceTex : sa.How, sa.Hi.X - sa.Lo.X, sa.Hi.Y - sa.Lo.Y, sa.Hi.Z - sa.Lo.Z, string.Join(" ", sa.Textures), string.Join(" ", sa.Detail)));
}
Log($"signs.jsonl: {nSigns} placements of {signs.Count} archetypes ({nSignsDlc} only in DLC packs' maps)");

// ---- painted words: the word cells of the marking atlases, mapped through each decal triangle's UVs into the world
var pieces = new List<Piece>();
using (var wordUv = surveyDir != null ? new StreamWriter(Path.Combine(surveyDir, "survey_word_uv.tsv")) : null)
  foreach (var pl in places)
  {
    if (!wordArch.Contains(pl.Arch)) continue;
    var a = arch[pl.Arch];
    var dr = LoadDrawable(pl.Arch, a);
    var models = dr?.DrawableModels?.High;
    if (models == null) continue;
    foreach (var m in models)
      foreach (var g in m.Geometries)
      {
        var diff = Diffuse(g.Shader);
        var cells = diff != null ? Words.Cells(diff) : null;
        if (cells == null) continue;
        SavePng(diff, dr, a.TextureDict.Hash);
        var vd = g.VertexData; var ib = g.IndexBuffer?.Indices;
        if (vd == null || ib == null) continue;
        int nv = vd.VertexCount;
        var P = new Vector3[nv]; var UV = new Vector2[nv];
        bool hasCol = (vd.Info.Flags & (1u << 4)) != 0;
        var A = new float[nv];  // vertex alpha: decals fade out with it
        for (int v = 0; v < nv; v++) { P[v] = pl.P + pl.Q.Multiply(vd.GetVector3(v, 0) * pl.Scale); UV[v] = vd.GetVector2(v, 6); A[v] = hasCol ? vd.GetColour(v, 4).A / 255f : 1; }
        // one painted word = the triangles of one cell that touch each other (vertices at the same place count as one)
        var vid = new int[nv]; var ids = new Dictionary<(int, int, int), int>();
        for (int v = 0; v < nv; v++) { var k = ((int)MathF.Round(P[v].X * 50), (int)MathF.Round(P[v].Y * 50), (int)MathF.Round(P[v].Z * 50)); if (!ids.TryGetValue(k, out vid[v])) ids[k] = vid[v] = ids.Count; }
        var parent = Enumerable.Range(0, ids.Count).ToArray();
        int Find(int x) { while (parent[x] != x) x = parent[x] = parent[parent[x]]; return x; }
        for (int i = 0; i + 2 < g.IndicesCount; i += 3) { parent[Find(vid[ib[i]])] = Find(vid[ib[i + 1]]); parent[Find(vid[ib[i + 1]])] = Find(vid[ib[i + 2]]); }
        var acc = new Dictionary<(int grp, int cell), Acc>();
        if (wordUv != null)
        {
          // how the map's decals cut this texture up: each connected patch's UV box
          var box = new Dictionary<int, float[]>();
          for (int i = 0; i + 2 < g.IndicesCount; i++)
          {
            int r = Find(vid[ib[i]]); var uv = UV[ib[i]];
            if (!box.TryGetValue(r, out var b)) box[r] = b = new[] { uv.X, uv.X, uv.Y, uv.Y };
            b[0] = MathF.Min(b[0], uv.X); b[1] = MathF.Max(b[1], uv.X); b[2] = MathF.Min(b[2], uv.Y); b[3] = MathF.Max(b[3], uv.Y);
          }
          foreach (var b in box.Values) wordUv.WriteLine(string.Format(inv, "{0}	{1}	{2:F3}	{3:F3}	{4:F3}	{5:F3}", diff, a.Name, b[0], b[1], b[2], b[3]));
        }
        for (int i = 0; i + 2 < g.IndicesCount; i += 3)
        {
          int i0 = ib[i], i1 = ib[i + 1], i2 = ib[i + 2];
          Vector2 a0 = UV[i0], a1 = UV[i1], a2 = UV[i2];
          float det = (a1.X - a0.X) * (a2.Y - a0.Y) - (a2.X - a0.X) * (a1.Y - a0.Y);
          if (MathF.Abs(det) < 1e-9f) continue;
          // world position as a function of (u, v) over this triangle: dP/du and dP/dv
          var e1 = P[i1] - P[i0]; var e2 = P[i2] - P[i0];
          var dPdu = (e1 * (a2.Y - a0.Y) - e2 * (a1.Y - a0.Y)) / det;
          var dPdv = (e2 * (a1.X - a0.X) - e1 * (a2.X - a0.X)) / det;
          var nrm = Vector3.Cross(e1, e2); if (nrm.Length() < 1e-9f) continue; nrm.Normalize();
          // decals are often mapped whole tiles away from the texture's 0..1 square: try every tile the triangle touches
          int tu0 = (int)MathF.Floor(MathF.Min(a0.X, MathF.Min(a1.X, a2.X))), tu1 = (int)MathF.Floor(MathF.Max(a0.X, MathF.Max(a1.X, a2.X)));
          int tv0 = (int)MathF.Floor(MathF.Min(a0.Y, MathF.Min(a1.Y, a2.Y))), tv1 = (int)MathF.Floor(MathF.Max(a0.Y, MathF.Max(a1.Y, a2.Y)));
          if (tu1 - tu0 > 16 || tv1 - tv0 > 16) continue;
          for (int tu = tu0; tu <= tu1; tu++)
            for (int tv = tv0; tv <= tv1; tv++)
              for (int ci = 0; ci < cells.Length; ci++)
              {
                var c = cells[ci];
                var poly = Clip(new List<Vector2> { a0, a1, a2 }, tu + c.U0, tu + c.U1, tv + c.V0, tv + c.V1);
                if (poly.Count < 3) continue;
                double ar = 0, cu = 0, cv = 0;
                for (int k = 0; k < poly.Count; k++)
                {
                  var q0 = poly[k]; var q1 = poly[(k + 1) % poly.Count];
                  double cr = (double)q0.X * q1.Y - (double)q1.X * q0.Y;
                  ar += cr; cu += (q0.X + q1.X) * cr; cv += (q0.Y + q1.Y) * cr;
                }
                if (Math.Abs(ar) < 1e-12) continue;
                cu /= 3 * ar; cv /= 3 * ar; ar = Math.Abs(ar) / 2;
                var wp = P[i0] + dPdu * (float)(cu - a0.X) + dPdv * (float)(cv - a0.Y);
                var key = (Find(vid[i0]), ci * 10000 + (tu - tu0) * 100 + (tv - tv0));
                if (!acc.TryGetValue(key, out var ac)) acc[key] = ac = new Acc { Cell = c };
                double wa = ar * Vector3.Cross(dPdu, dPdv).Length();  // world area of the piece
                ac.Alpha += (A[i0] + A[i1] + A[i2]) / 3 * ar;
                ac.UvArea += ar; ac.WorldArea += wa; ac.C += wp * (float)wa; ac.Du += dPdu * (float)ar; ac.Dv += dPdv * (float)ar; ac.Nz += MathF.Abs(nrm.Z) * (float)ar;
                ac.Mirror += (Vector3.Dot(Vector3.Cross(dPdu, dPdv), new Vector3(0, 0, 1)) > 0 ? 1 : -1) * ar;
              }
        }
        foreach (var ac in acc.Values)
        {
          if (ac.Nz / ac.UvArea < 0.7) continue;  // on a wall, not on the ground
          pieces.Add(new Piece { Cell = ac.Cell, Tex = diff, Ent = a.Name, Src = pl.Src, UvArea = ac.UvArea, WorldArea = ac.WorldArea, Mirror = ac.Mirror, Alpha = ac.Alpha, C = ac.C, Du = ac.Du, Dv = ac.Dv });
        }
      }
  }
// A word cut across two road sections (or two patches of one) comes in pieces: join the pieces of the same cell that lie
// within a word's size of each other and together map no more than the one cell, the nearest first. Slivers don't
// count: a line strip that strays into a word's cell is not a piece of the word.
double Cover(Piece q) => q.UvArea / ((q.Cell.U1 - q.Cell.U0) * (q.Cell.V1 - q.Cell.V0));
var near = new List<(float d, int a, int b)>();
foreach (var grp in Enumerable.Range(0, pieces.Count).GroupBy(n => (pieces[n].Tex, pieces[n].Cell)))
{
  var idx = grp.Where(n => Cover(pieces[n]) >= 0.15).ToArray();
  for (int x = 0; x < idx.Length; x++)
    for (int y = x + 1; y < idx.Length; y++)
    {
      float d = (pieces[idx[x]].C / (float)pieces[idx[x]].WorldArea - pieces[idx[y]].C / (float)pieces[idx[y]].WorldArea).Length();
      if (d < 4) near.Add((d, idx[x], idx[y]));
    }
}
var into = Enumerable.Range(0, pieces.Count).ToArray();
int Root(int x) { while (into[x] != x) x = into[x] = into[into[x]]; return x; }
foreach (var (d, x0, y0) in near.OrderBy(n => n.d))
{
  int x = Root(x0), y = Root(y0);
  if (x == y) continue;
  Piece px = pieces[x], py = pieces[y];
  if (Cover(px) + Cover(py) > 1.15 || Vector3.Dot(Vector3.Normalize(px.Dv), Vector3.Normalize(py.Dv)) < 0.9f) continue;
  px.UvArea += py.UvArea; px.WorldArea += py.WorldArea; px.Mirror += py.Mirror; px.Alpha += py.Alpha; px.C += py.C; px.Du += py.Du; px.Dv += py.Dv; px.Parts += py.Parts;
  py.Gone = true; into[y] = x;
}
var whole = pieces.Where(q => !q.Gone).ToList();
// The STOP of the STOP AHEAD texture is painted on its own at most junctions. Where its AHEAD is painted too (some tens
// of metres along the same lane), that STOP is the warning's, not a stop line's.
var stopAhead = new HashSet<Piece>();
foreach (var ah in whole.Where(q => q.Cell.Word == "AHEAD" && Cover(q) >= 0.5))
{
  var stop = whole.Where(q => q.Cell.Word == "STOP" && q.Tex == ah.Tex && Cover(q) >= 0.5 && Vector3.Dot(Vector3.Normalize(q.Dv), Vector3.Normalize(ah.Dv)) > 0.9f)
    .OrderBy(q => (q.C / (float)q.WorldArea - ah.C / (float)ah.WorldArea).Length()).FirstOrDefault();
  if (stop != null && (stop.C / (float)stop.WorldArea - ah.C / (float)ah.WorldArea).Length() < 45) stopAhead.Add(stop);
}
int nWords = 0, nPartial = 0, nFaded = 0;
var wordCounts = new SortedDictionary<string, int>();
using (var partial = surveyDir != null ? new StreamWriter(Path.Combine(surveyDir, "survey_partial_words.jsonl")) : null)
using (var w = new StreamWriter(Path.Combine(outDir, "words.jsonl")))
  foreach (var q in whole)
  {
    var c = q.Cell;
    double cover = Cover(q), alpha = q.Alpha / q.UvArea;
    var ctr = q.C / (float)q.WorldArea;
    var du = q.Du / (float)q.UvArea; var dv = q.Dv / (float)q.UvArea;
    // the top of the text in the world: where the cell's "up" axis points
    var up = c.Up switch { "-v" => -dv, "+v" => dv, "+u" => du, _ => -du };
    var across = c.Up[1] == 'v' ? du : dv;
    float len = up.Length() * (c.Up[1] == 'v' ? c.V1 - c.V0 : c.U1 - c.U0), wid = across.Length() * (c.Up[1] == 'v' ? c.U1 - c.U0 : c.V1 - c.V0);
    // seen from above, text reads left to right only for one handedness of the UV mapping (v runs down the texture)
    bool mirrored = q.Mirror > 0;
    var word = stopAhead.Contains(q) ? "STOP AHEAD" : c.Word;
    bool keep = cover >= 0.5 && alpha >= 0.25;
    if (cover < 0.5) nPartial++; else if (!keep) nFaded++;
    var o2 = keep ? w : partial;
    if (o2 == null) continue;
    o2.WriteLine(string.Format(inv, "{{\"word\":\"{0}\",\"x\":{1:F3},\"y\":{2:F3},\"z\":{3:F3},\"heading\":{4:F1},\"tex\":\"{5}\",\"ent\":\"{6}\",\"cover\":{7:F2},\"length\":{8:F2},\"width\":{9:F2}{10}{11}{12}{13},\"ymap\":\"{14}\"}}",
      word, ctr.X, ctr.Y, ctr.Z, Math.Atan2(up.Y, up.X) * 180 / Math.PI, q.Tex, q.Ent, cover, len, wid, alpha < 0.95 ? string.Format(inv, ",\"alpha\":{0:F2}", alpha) : "",
      q.Parts > 1 ? $",\"pieces\":{q.Parts}" : "", mirrored ? ",\"mirrored\":true" : "", Dlc(q.Src) ? ",\"dlc\":true" : "", q.Src));
    if (keep) { nWords++; wordCounts[word] = wordCounts.GetValueOrDefault(word) + 1; }
  }
Log($"words.jsonl: {nWords} words ({string.Join(", ", wordCounts.Select(k => $"{k.Key}={k.Value}"))}); {nPartial} partly mapped cells and {nFaded} faded-out words dropped");
return 0;

// the part of a polygon inside the rectangle u0..u1 x v0..v1
static List<Vector2> Clip(List<Vector2> poly, float u0, float u1, float v0, float v1)
{
  for (int side = 0; side < 4 && poly.Count > 0; side++)
  {
    float D(Vector2 p) => side switch { 0 => p.X - u0, 1 => u1 - p.X, 2 => p.Y - v0, _ => v1 - p.Y };
    var o = new List<Vector2>();
    for (int k = 0; k < poly.Count; k++)
    {
      var a = poly[k]; var b = poly[(k + 1) % poly.Count];
      float da = D(a), db = D(b);
      if (da >= 0) o.Add(a);
      if (da >= 0 != db >= 0) o.Add(a + (b - a) * (da / (da - db)));
    }
    poly = o;
  }
  return poly;
}

// only the vanilla archives: skip mod folders (e.g. a disabled NaturalVision install) that sit beside them
static bool Vanilla(string p) { var l = p.ToLowerInvariant(); return l.StartsWith("x64") || l.StartsWith("update\\") || l.StartsWith("common"); }

// which archive's copy of a file the game ends up using: update.rpf over DLC packs (later patchdays over earlier, over
// other packs) over the base archives
static int Priority(string p)
{
  var l = p.ToLowerInvariant();
  if (l.StartsWith("update\\update.rpf")) return 100000;
  var m = Regex.Match(l, @"dlcpacks\\patchday(\d+)");
  if (m.Success) return 1000 + int.Parse(m.Groups[1].Value);
  if (l.Contains("dlcpacks\\")) return 500;
  return 0;
}

// a DLC pack's own map, as opposed to the base map and the patchday packs that patch it
static bool Dlc(string p) { var l = p.ToLowerInvariant().Replace('\\', '/'); return l.Contains("/dlc_patch/") || Regex.IsMatch(l, "dlcpacks/(?!patchday)"); }

class Place { public uint Arch; public Vector3 P; public Quaternion Q; public Vector3 Scale; public string Src; public int Dups; }
class SignArch { public string Name, FaceTex, How, Says = ""; public int Count; public Vector3 Face, FaceCentre, Lo, Hi; public List<Vector3> Faces = new(); public List<string> Textures = new(); public List<string> Detail = new(); }
class FaceAcc
{
  public int Rank; public double Area; public double[] Oct = new double[8]; public Vector3[] N = new Vector3[8], C = new Vector3[8];
  public string Describe(IFormatProvider inv) => Oct.Sum() <= 0 ? "none" : string.Join("/", Enumerable.Range(0, 8).Where(o => Oct[o] > 0.02 * Area).OrderByDescending(o => Oct[o]).Select(o => string.Format(inv, "{0}deg:{1:F2}", o * 45, Oct[o])));
}
class Acc { public Words.Cell Cell; public double UvArea, WorldArea, Mirror, Alpha; public Vector3 C, Du, Dv; public float Nz; }
class Piece { public Words.Cell Cell; public string Tex, Ent, Src; public double UvArea, WorldArea, Mirror, Alpha; public Vector3 C, Du, Dv; public int Parts = 1; public bool Gone; }

// Painted words: texture (exact name, lower case) -> the cells of it that are one word each. Up is the texture axis that
// points at the top of the text.
static class Words
{
  public record Cell(string Word, float U0, float U1, float V0, float V1, string Up);
  // keys are texture name prefixes (the longest matching one applies): the same picture ships under several names
  // (diffuse + alpha map names run together, _big / _lod variants). One cell per word, as the map's decals use them:
  // KEEP and CLEAR, PED and XING, STAY, IN and LANE are laid down as separate patches. Cell edges are the text's own,
  // measured from the textures' alpha, except the car-park atlases', which are roadpaint's atlas_cells.json cells.
  static readonly (string key, Cell[] cells)[] table =
  {
    ("im_roadmarkings07", new[] { new Cell("STOP", .045f, .985f, .02f, .477f, "-v") }),
    ("rsn_et_keepclearrsn_et_keepclear", new[] { new Cell("KEEP", .012f, .98f, .012f, .238f, "-v"), new Cell("CLEAR", .012f, .98f, .262f, .488f, "-v"), new Cell("STOP", .012f, .98f, .625f, .988f, "-v") }),
    ("vb1_21_rsn_rb_redwallrsn_et_keepclear", new[] { new Cell("KEEP", .012f, .98f, .012f, .238f, "-v"), new Cell("CLEAR", .012f, .98f, .262f, .488f, "-v"), new Cell("STOP", .012f, .98f, .625f, .988f, "-v") }),
    ("cs_rsn_sl_road_0009", new[] { new Cell("AHEAD", 0f, .56f, 0f, .78f, "-v"), new Cell("STOP", .6f, 1f, 0f, .78f, "-v") }),
    ("im_roadmarkings08", new[] { new Cell("WAIT", .012f, .555f, .012f, .465f, "-v"), new Cell("HERE", .57f, .992f, .012f, .465f, "-v"), new Cell("PED", .016f, .465f, .496f, .992f, "-v"), new Cell("XING", .492f, .98f, .496f, .992f, "-v") }),
    ("im_roadmarkings09", new[] { new Cell("ONLY", .012f, .977f, .012f, .977f, "-v") }),
    ("im_roadmarkings10", new[] { new Cell("STAY", .016f, .68f, .01f, .467f, "-v"), new Cell("IN", .727f, .967f, .01f, .467f, "-v"), new Cell("LANE", .02f, .789f, .498f, .992f, "-v") }),
    ("im_roadmarkings11", new[] { new Cell("KEEP", .094f, .97f, .01f, .379f, "-v"), new Cell("CLEAR", .004f, .994f, .387f, .75f, "-v"), new Cell("NO PARKING", .004f, .994f, .773f, .99f, "-v") }),
    ("et_numbers3et_keepclear", new[] { new Cell("KEEP", .125f, .875f, .109f, .422f, "-v"), new Cell("CLEAR", .023f, .969f, .586f, .898f, "-v") }),
    ("rsn_os_marking01", new[] { new Cell("KEEP", .004f, .418f, 0f, 1f, "-v"), new Cell("CLEAR", .438f, 1f, 0f, 1f, "-v") }),
    ("dt1_23_keepout_marking01", new[] { new Cell("KEEP", .002f, .418f, 0f, 1f, "-v"), new Cell("CLEAR", .439f, 1f, 0f, 1f, "-v") }),
    ("ap1_02_taxi", new[] { new Cell("TAXI", .021f, .965f, .211f, .781f, "-v") }),
    ("gg_dockfloor_text", new[] {
      new Cell("AHEAD", .059f, .477f, .051f, .191f, "-v"), new Cell("BUMP", .059f, .477f, .242f, .387f, "-v"), new Cell("RAIL", .059f, .477f, .438f, .578f, "-v"),
      new Cell("AREA", .059f, .477f, .613f, .754f, "-v"), new Cell("EAST", .059f, .477f, .805f, .945f, "-v"),
      new Cell("DOCKS", .512f, .961f, .051f, .191f, "-v"), new Cell("SOUTH", .512f, .961f, .242f, .387f, "-v"), new Cell("ONLY", .512f, .961f, .438f, .578f, "-v"),
      new Cell("ENTRY", .512f, .961f, .613f, .754f, "-v"), new Cell("EXIT", .512f, .961f, .805f, .945f, "-v") }),
    ("rnos_roadmarkings_001", new[] { new Cell("EXIT", .4f, 1f, 0f, .4f, "-v"), new Cell("IN", .65f, 1f, .4f, 1f, "-v") }),
    ("rnos_roadmarkings_002", new[] { new Cell("EXIT", .4f, 1f, 0f, .3f, "-v"), new Cell("IN", .6f, 1f, .3f, .7f, "-v"), new Cell("KEEP", .012f, .451f, .7f, .85f, "-v"), new Cell("CLEAR", .469f, .992f, .7f, .85f, "-v"), new Cell("NO PARKING", .02f, .988f, .85f, 1f, "-v") }),
    ("im_ggroadmarkings01", new[] { new Cell("EXIT", .53f, .85f, .5f, .75f, "-v") }),
    ("im_carparkmarkings01", new[] { new Cell("EXIT", .55f, .82f, .5f, 1f, "-v") }),
    ("bh1_13_roadmarkings", new[] { new Cell("EXIT", .2f, .45f, .45f, 1f, "-u") }),
    ("bh1_15_genallmarkings", new[] { new Cell("EXIT", .2f, .45f, .45f, 1f, "-u") }),
    ("sc1_30_arrows", new[] { new Cell("EXIT", .65f, 1f, .2f, 1f, "+u") }),
    ("sc1_32_cmk_roadmarkings_001", new[] { new Cell("EXIT", .4f, 1f, 0f, .45f, "-v"), new Cell("IN", .6f, 1f, .45f, 1f, "-v") }),
  };
  static readonly Dictionary<string, Cell[]> cache = new();
  public static Cell[] Cells(string tex)
  {
    if (cache.TryGetValue(tex, out var c)) return c;
    string best = null;
    foreach (var (key, cells) in table) if (tex.StartsWith(key) && (best == null || key.Length > best.Length)) { best = key; c = cells; }
    return cache[tex] = c;
  }
}

// What each sign face texture says, from its name and from looking at it.
static class Says
{
  static readonly Dictionary<string, string> table = new()
  {
    ["prop_sign_road_stop_01"] = "stop", ["prop_sign_road_stop_01_snw"] = "stop", ["prop_sign_road_stopahead_01"] = "stop ahead",
    ["prop_sign_road_yield_01"] = "yield", ["prop_sign_road_donotenter_01"] = "do not enter", ["prop_sign_road_wrongway_01"] = "wrong way",
    ["prop_sign_road_oneway_01"] = "one way", ["prop_sign_road_oneway_02"] = "one way",
    ["prop_sign_road_keepright_01"] = "keep right except to pass", ["prop_sign_road_keepright_02"] = "keep right",
    ["prop_sign_road_noturn_01"] = "no left turn", ["prop_sign_road_nouturn_01"] = "no u-turn", ["prop_sign_road_alltraffic_01"] = "all traffic (arrow)",
    ["prop_sign_road_intersection_01"] = "do not block intersection", ["prop_sign_road_exit"] = "exit only", ["prop_sign_road_hospital_01"] = "hospital",
    ["prop_sign_speed_5"] = "speed limit 5", ["prop_sign_speed_15"] = "speed limit 15", ["prop_sign_speed_25"] = "speed limit 25", ["prop_sign_speed_35"] = "speed limit 35",
    ["prop_sign_speed_35_snw"] = "speed limit 35", ["prop_sign_speed_50"] = "speed limit 50", ["prop_sign_speed_60"] = "speed limit 60",
    ["prop_sign_speed_exit_25"] = "exit 25 mph (advisory)", ["prop_sign_speed_exit_55"] = "exit 55 mph (advisory)", ["prop_sign_speed_ramp"] = "ramp 35 mph (advisory)",
    ["prop_sign_speed_trucks_5"] = "trucks speed limit 5", ["prop_sign_speed_trucks_20"] = "trucks speed limit 20", ["prop_sign_speed_trucks_45"] = "trucks speed limit 45",
    ["prop_sign_speed_trucks_55"] = "autos with trailers and trucks 55 maximum", ["prop_sign_speed_aircraft"] = "speed enforced by aircraft",
    ["prop_sign_road_45arrow_01"] = "warning: curve", ["prop_sign_road_90arrow_01"] = "warning: sharp turn", ["prop_sign_road_windingroad_01"] = "warning: winding road",
    ["prop_sign_road_windingroad_02"] = "warning: winding road", ["prop_sign_road_cheveron_01"] = "chevron (bend marker)", ["prop_sign_road_straightarrow_01"] = "direction arrow (wide)",
    ["prop_sign_road_straightarrow_02"] = "direction arrow (narrow)", ["prop_sign_road_splitlane_01"] = "warning: two-direction arrows", ["prop_sign_road_lane_01"] = "warning: lane ends",
    ["prop_sign_road_merge_01"] = "warning: merge", ["prop_sign_road_merge_02"] = "warning: merge (arrow)", ["prop_sign_road_pedxing_01"] = "warning: pedestrian crossing",
    ["prop_sign_road_pedxing_02"] = "warning: school crossing", ["prop_sign_road_hiddenintersection_01"] = "warning: hidden intersection", ["prop_sign_road_notathroughroad_01"] = "not a through street",
    ["prop_sign_road_slippery_01"] = "warning: slippery road", ["prop_sign_road_fallingrocks_01"] = "warning: falling rocks", ["prop_sign_road_flooding_01"] = "warning: subject to flooding",
    ["prop_sign_road_cow_01"] = "warning: cattle", ["prop_sign_road_deer_01"] = "warning: deer", ["prop_sign_road_horse_01"] = "warning: horse riders", ["prop_sign_road_tractor_01"] = "warning: tractors",
    ["prop_sign_road_tram_01"] = "warning: trams, look both ways", ["prop_sign_road_truckentrance_01"] = "warning: truck entrance", ["prop_sign_road_lowaircraft_01"] = "warning: low flying aircraft",
    ["prop_sign_road_pier_01"] = "warning: pier ends", ["prop_sign_road_railroad_01"] = "warning: railroad crossing", ["prop_sign_road_railroad_02"] = "2 tracks",
    ["prop_sign_road_roadworkahead_01"] = "road work ahead", ["prop_sign_road_menatwork_01"] = "men at work", ["prop_sign_road_unevenpavement_01"] = "uneven pavement",
    ["prop_sign_road_stoppedvehicles_01"] = "watch for stopped vehicles", ["prop_sign_road_warning"] = "hazard stripes (object marker)", ["prop_sign_road_reflector_01"] = "red reflectors (object marker)",
    ["prop_sign_road_usecrosswalk_01"] = "no pedestrian crossing, use crosswalk", ["prop_sign_road_bike_01"] = "bike route", ["prop_sign_road_bike_04"] = "road closed to autos, bicycles permitted",
    ["prop_sign_road_parking_01"] = "no standing anytime", ["prop_sign_road_parking_03"] = "no stopping 9pm-6am", ["prop_sign_road_parking_09"] = "2 hour parking", ["prop_sign_road_parking_12"] = "no parking (times)",
    ["prop_sign_road_parking_14"] = "no standing, tow away", ["prop_sign_road_parking_16"] = "no parking any time", ["prop_sign_road_parking_17"] = "no parking, tow zone",
    ["prop_sign_road_litter_02"] = "unlawful to litter", ["prop_sign_road_loitering_01"] = "respect the neighborhood (quiet)", ["prop_sign_road_loitering_02"] = "no loitering",
    ["prop_sign_road_restriction_05"] = "no loud stereos or horn", ["prop_sign_road_restriction_06"] = "private property, no trespassing", ["prop_sign_road_restriction_10"] = "no trespassing",
    ["prop_sign_road_security_01"] = "security company (Bobcat)", ["prop_sign_road_security_02"] = "security company (Gruppe Sechs)", ["prop_sign_road_security_03"] = "neighborhood watch",
    ["prop_sign_road_security_04"] = "LSPD video monitoring", ["prop_sign_road_security_05"] = "no drugs or alcohol", ["prop_sign_road_security_06"] = "neighborhood crime watch",
    ["prop_sign_road_smoking_01"] = "danger, no smoking", ["prop_sign_road_wildlife_01"] = "caution: dangerous animals", ["prop_sign_road_wildlife_02"] = "caution: rattlesnakes",
    ["prop_sign_road_wildlife_03"] = "do not feed the wildlife", ["prop_sign_road_dike_01"] = "keep off dike", ["prop_sign_road_dog_01"] = "dogs on leash", ["prop_sign_road_dog_02"] = "no dogs allowed",
    ["prop_sign_road_park_01"] = "park rules", ["prop_sign_road_park_03"] = "park closed at night", ["prop_sign_road_beach_01"] = "beach rules", ["prop_sign_road_beach_02"] = "beach danger notice",
    ["prop_sign_road_beach_03"] = "lifeguards on duty", ["prop_sign_road_beach_05"] = "no swimming, no wading", ["prop_sign_road_beach_06"] = "no wading or boats", ["prop_sign_road_beach_07"] = "no swimming",
    ["prop_sign_road_airport_blue"] = "airport direction board", ["prop_sign_road_highwayfront"] = "freeway guide sign board",
    ["fwy_rsn_mjg_signblank_001"] = "freeway guide sign board", ["prop_sign_freewayentrance"] = "freeway entrance", ["prop_sign_hitch_hike"] = "prison area, do not pick up hitch hikers",
    ["prop_sign_interstate_01prop_sign_interstate_a"] = "interstate 1 shield", ["prop_sign_interstate_02prop_sign_interstate_a"] = "interstate 2 shield",
    ["prop_sign_interstate_04prop_sign_interstate_a"] = "interstate 4 shield", ["prop_sign_interstate_05prop_sign_interstate_a"] = "interstate 5 shield",
    ["prop_sign_route_01prop_sign_route_a"] = "US route 1 shield", ["prop_sign_route_13prop_sign_route_a"] = "US route 13 shield", ["prop_sign_route_15prop_sign_route_a"] = "US route 15 shield",
    ["prop_sign_callbox"] = "call box", ["prop_sign_loading_1"] = "loading only", ["prop_sign_parking_1"] = "caution, vehicle exiting", ["prop_sign_taxi_1"] = "taxi zone", ["prop_busstop_sign"] = "bus stop",
    ["prop_traffic_01a"] = "traffic light", ["prop_traffic_railcrossing"] = "railroad crossing signal", ["prop_traffic_rail_3"] = "railroad crossing crossbuck + stop",
  };
  public static string OfTexture(string tex) => table.GetValueOrDefault(tex);
  static readonly Dictionary<string, string> byArchetype = new()
  {
    ["prop_traffic_03b"] = "pedestrian signal",  // a pole with only the walk / don't walk box
  };
  public static string OfArchetype(string arch) => byArchetype.GetValueOrDefault(arch);
  // Where one texture holds the whole model, the parts of it that tell which way the model faces, the most telling
  // first. The traffic light's lens plate is no use: it is modelled two-sided. Its heads' ribbed back panel faces away
  // from the drivers (reversed); the pedestrian box's LED grille faces the walkers.
  public static (string what, float u0, float u1, float v0, float v1, bool reversed)[] FaceParts(string tex) => tex switch
  {
    "prop_traffic_01a" => new[] { ("head back panel, reversed", 0.62f, 0.92f, 0.32f, 0.43f, true), ("pedestrian grille", 0.5f, 1f, 0.78f, 1f, false) },
    "prop_traffic_railcrossing" => new[] { ("RAIL ROAD CROSSING boards", 0.31f, 1f, 0f, 0.23f, false) },
    _ => null,
  };
}

static class Png
{
  // 8-bit RGBA PNG from CodeWalker's BGRA pixels
  public static void Write(string path, byte[] bgra, int w, int h)
  {
    var raw = new byte[h * (w * 4 + 1)];
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
      {
        int s = (y * w + x) * 4, d = y * (w * 4 + 1) + 1 + x * 4;
        raw[d] = bgra[s + 2]; raw[d + 1] = bgra[s + 1]; raw[d + 2] = bgra[s]; raw[d + 3] = bgra[s + 3];
      }
    using var ms = new MemoryStream();
    using (var z = new ZLibStream(ms, CompressionLevel.Fastest, true)) z.Write(raw);
    using var f = File.Create(path);
    f.Write(new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 });
    var hdr = new byte[13];
    hdr[0] = (byte)(w >> 24); hdr[1] = (byte)(w >> 16); hdr[2] = (byte)(w >> 8); hdr[3] = (byte)w;
    hdr[4] = (byte)(h >> 24); hdr[5] = (byte)(h >> 16); hdr[6] = (byte)(h >> 8); hdr[7] = (byte)h;
    hdr[8] = 8; hdr[9] = 6;
    Chunk(f, "IHDR", hdr); Chunk(f, "IDAT", ms.ToArray()); Chunk(f, "IEND", Array.Empty<byte>());
  }
  static void Chunk(Stream f, string type, byte[] data)
  {
    var t = Encoding.ASCII.GetBytes(type);
    f.Write(new byte[] { (byte)(data.Length >> 24), (byte)(data.Length >> 16), (byte)(data.Length >> 8), (byte)data.Length });
    f.Write(t); f.Write(data);
    uint c = 0xFFFFFFFF;
    foreach (var b in t.Concat(data)) { c ^= b; for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320 & (uint)-(int)(c & 1)); }
    c ^= 0xFFFFFFFF;
    f.Write(new byte[] { (byte)(c >> 24), (byte)(c >> 16), (byte)(c >> 8), (byte)c });
  }
}
