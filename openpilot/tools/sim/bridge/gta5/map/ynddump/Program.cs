// Dumps GTA V's path nodes (.ynd) and their street names from the game's archives as JSON lines, one node ("n"), link ("l")
// or street ("s") each: ynddump <game folder> <out.jsonl> [<minimap.jsonl>]. With a third file, also GTA's minimap road
// art (ynd_to_osm.py --minimap). It only reads the game's files.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using CodeWalker.GameFiles;

if (args.Length is < 2 or > 3)
{
  Console.Error.WriteLine("usage: ynddump <game folder> <out.jsonl> [<minimap.jsonl>]");
  return 1;
}
var (game, outPath, minimapPath) = (args[0], args[1], args.Length > 2 ? args[2] : null);
GTA5Keys.LoadFromPath(game, true, null);
var rpf = new RpfManager();
rpf.Init(game, true, s => { }, e => Console.Error.WriteLine(e));

// later archives (update, patches) override earlier ones by name
var ynds = new Dictionary<string, RpfFileEntry>();
foreach (var r in rpf.AllRpfs)
  foreach (var e in r.AllEntries)
    if (e is RpfFileEntry fe && e.NameLower.EndsWith(".ynd"))
      ynds[e.NameLower] = fe;
Console.Error.WriteLine($"{ynds.Count} ynd files");

using var w = new StreamWriter(outPath);
int nn = 0, nl = 0;
var streets = new HashSet<uint>();
var inv = System.Globalization.CultureInfo.InvariantCulture;
foreach (var (name, fe) in ynds.OrderBy(k => k.Key))
{
  var y = rpf.GetFile<YndFile>(fe);
  if (y?.Nodes == null) continue;
  foreach (var n in y.Nodes)
  {
    var r = n.RawData;
    w.WriteLine(string.Format(inv, "{{\"t\":\"n\",\"a\":{0},\"i\":{1},\"x\":{2:F2},\"y\":{3:F2},\"z\":{4:F2},\"f\":[{5},{6},{7},{8},{9}],\"st\":{10},\"sp\":{11}}}",
      n.AreaID, n.NodeID, n.Position.X, n.Position.Y, n.Position.Z,
      r.Flags0.Value, r.Flags1.Value, r.Flags2.Value, r.Flags3.Value, r.Flags4.Value, r.StreetName.Hash, (int)n.Speed));
    nn++;
    if (r.StreetName.Hash != 0) streets.Add(r.StreetName.Hash);
    for (int k = 0; k < n.LinkCount; k++)
    {
      var lr = y.NodeDictionary.Links[n.LinkID + k];
      w.WriteLine($"{{\"t\":\"l\",\"a\":{n.AreaID},\"i\":{n.NodeID},\"ta\":{lr.AreaID},\"ti\":{lr.NodeID},\"f\":[{lr.Flags0.Value},{lr.Flags1.Value},{lr.Flags2.Value}],\"len\":{lr.LinkLength.Value}}}");
      nl++;
    }
  }
}
// street names: the nodes' street hashes are the keys of the game's (American English) text
var names = new Dictionary<uint, string>();
foreach (var r in rpf.AllRpfs)
  foreach (var e in r.AllEntries)
    if (e is RpfFileEntry fe && e.NameLower.EndsWith(".gxt2") && (e.Path.Contains("american_rel") || e.Path.Contains("americandlc.rpf") || e.Path.Contains("american.rpf")))
    {
      var g = rpf.GetFile<Gxt2File>(fe);
      if (g?.TextEntries == null) continue;
      foreach (var t in g.TextEntries)
        if (streets.Contains(t.Hash)) names[t.Hash] = t.Text;
    }
foreach (var (h, name) in names.OrderBy(k => k.Key))
  w.WriteLine($"{{\"t\":\"s\",\"h\":{h},\"name\":{System.Text.Json.JsonSerializer.Serialize(name)}}}");
Console.Error.WriteLine($"{nn} nodes, {nl} links, {names.Count} of {streets.Count} street names -> {outPath}");
if (minimapPath == null)
  return 0;

// The minimap's roads are flat-coloured vector meshes in world x, y, with the draw layer as z. Their layer's grey
// triangles, one line per geometry and grey: 180 roads, 99 dirt tracks and alleys, 220 rail.
const float RoadLayer = 11.9f;
var minimap = new Regex(@"minimap[.]rpf.minimap_[0-9_]+[.]ydd$", RegexOptions.IgnoreCase);
var ydds = new Dictionary<string, RpfFileEntry>();
foreach (var r in rpf.AllRpfs)
  foreach (var e in r.AllEntries)
    if (e is RpfFileEntry fe && minimap.IsMatch(e.Path))
      ydds[e.NameLower] = fe;
using var mw = new StreamWriter(minimapPath);
int nt = 0;
foreach (var (_, fe) in ydds.OrderBy(k => k.Key))
{
  var y = rpf.GetFile<YddFile>(fe);
  foreach (var dr in y?.Drawables ?? [])
    foreach (var model in dr.AllModels ?? [])
      foreach (var g in model.Geometries ?? [])
      {
        var (vd, idx) = (g.VertexData, g.IndexBuffer?.Indices);
        if (vd == null || idx == null)
          continue;
        var byGrey = new SortedDictionary<int, StringBuilder>();
        for (int t = 0; t + 2 < idx.Length; t += 3)
        {
          var c = vd.GetColour(idx[t], 4);  // a triangle is its first vertex's colour
          if (vd.GetVector3(idx[t], 0).Z < RoadLayer || c.R != c.G || c.G != c.B)
            continue;
          if (!byGrey.TryGetValue(c.R, out var sb))
            byGrey[c.R] = sb = new StringBuilder();
          for (int k = 0; k < 3; k++)
          {
            var p = vd.GetVector3(idx[t + k], 0);
            sb.Append(sb.Length > 0 ? "," : "").Append(p.X.ToString("F1", inv)).Append(',').Append(p.Y.ToString("F1", inv));
          }
          nt++;
        }
        foreach (var (grey, sb) in byGrey)
          mw.WriteLine($"{{\"grey\":{grey},\"xy\":[{sb}]}}");
      }
}
Console.Error.WriteLine($"{nt} minimap road triangles from {ydds.Count} files -> {minimapPath}");
return 0;
