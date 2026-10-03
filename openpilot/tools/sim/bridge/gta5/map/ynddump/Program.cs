// Dumps GTA V's path nodes (.ynd) and their street names from the game's archives as JSON lines, one node ("n"), link ("l")
// or street ("s") each: ynddump <game folder> <out.jsonl>. It only reads the game's files.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CodeWalker.GameFiles;

if (args.Length != 2)
{
  Console.Error.WriteLine("usage: ynddump <game folder> <out.jsonl>");
  return 1;
}
var (game, outPath) = (args[0], args[1]);
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
    w.WriteLine(string.Format(inv, "{{\"t\":\"n\",\"a\":{0},\"i\":{1},\"x\":{2:F2},\"y\":{3:F2},\"z\":{4:F2},\"f\":[{5},{6},{7},{8},{9}],\"st\":{10}}}",
      n.AreaID, n.NodeID, n.Position.X, n.Position.Y, n.Position.Z,
      r.Flags0.Value, r.Flags1.Value, r.Flags2.Value, r.Flags3.Value, r.Flags4.Value, r.StreetName.Hash));
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
return 0;
