// ybnprobe <game> <out.jsonl> <x> <y> <radius>: static collision triangles near (x,y) with their material names, one JSON
// line each: {"m": name, "p": [x1,y1,z1,x2,y2,z2,x3,y3,z3]}. Only reads the game's files.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CodeWalker.GameFiles;
using SharpDX;

var game = args[0]; var outPath = args[1];
float qx = float.Parse(args[2]), qy = float.Parse(args[3]), qr = float.Parse(args[4]);
var inv = System.Globalization.CultureInfo.InvariantCulture;
GTA5Keys.LoadFromPath(game, true, null);
var rpf = new RpfManager();
rpf.Init(game, true, s => { }, e => Console.Error.WriteLine(e));
var mats = new List<string>();
var txt = rpf.GetFileUTF8Text("update\\update.rpf\\common\\data\\materials\\materials.dat") ?? rpf.GetFileUTF8Text("common.rpf\\data\\materials\\materials.dat");
foreach (var line in txt.Split('\n'))
{
  var l = line.Trim();
  if (l.Length == 0 || l[0] == '#' || l.StartsWith("Version", StringComparison.OrdinalIgnoreCase)) continue;
  var name = l.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries)[0];
  if (char.IsDigit(name[0])) continue;
  mats.Add(name);
}
Console.Error.WriteLine($"{mats.Count} materials, e.g. {string.Join(" ", mats.Where(m => m.Contains("TARMAC") || m.Contains("PAINT")))}");

var ybns = new Dictionary<string, RpfFileEntry>();
foreach (var r in rpf.AllRpfs)
{
  var lp = r.Path.ToLowerInvariant();
  if (!(lp.StartsWith("x64") || lp.StartsWith("update\\"))) continue;
  foreach (var e in r.AllEntries)
    if (e is RpfFileEntry fe && e.NameLower.EndsWith(".ybn") && !e.NameLower.StartsWith("hi@")) ybns[e.NameLower] = fe;
}
using var w = new StreamWriter(outPath);
var counts = new Dictionary<string, int>();
int files = 0;
foreach (var (name, fe) in ybns)
{
  YbnFile y;
  try { y = rpf.GetFile<YbnFile>(fe); } catch { continue; }
  var b = y?.Bounds;
  if (b == null) continue;
  if (b.BoxMax.X < qx - qr || b.BoxMin.X > qx + qr || b.BoxMax.Y < qy - qr || b.BoxMin.Y > qy + qr) continue;
  files++;
  var geoms = new List<BoundGeometry>();
  void Walk(Bounds bb) { if (bb is BoundComposite c) { if (c.Children?.data_items != null) foreach (var ch in c.Children.data_items) if (ch != null) Walk(ch); } else if (bb is BoundGeometry g) geoms.Add(g); }
  Walk(b);
  foreach (var g in geoms)
  {
    if (g.Polygons == null) continue;
    for (int i = 0; i < g.Polygons.Length; i++)
    {
      if (g.Polygons[i] is not BoundPolygonTriangle t) continue;
      var p1 = t.Vertex1; var p2 = t.Vertex2; var p3 = t.Vertex3;
      var cx = (p1.X + p2.X + p3.X) / 3; var cy = (p1.Y + p2.Y + p3.Y) / 3;
      if (Math.Abs(cx - qx) > qr || Math.Abs(cy - qy) > qr) continue;
      var mi = g.GetMaterial(i).Type.Index;
      var mn = mi < mats.Count ? mats[mi] : $"#{mi}";
      counts[mn] = counts.GetValueOrDefault(mn) + 1;
      w.WriteLine(string.Format(inv, "{{\"f\":\"{0}\",\"m\":\"{1}\",\"p\":[{2:F3},{3:F3},{4:F3},{5:F3},{6:F3},{7:F3},{8:F3},{9:F3},{10:F3}]}}", name, mn, p1.X, p1.Y, p1.Z, p2.X, p2.Y, p2.Z, p3.X, p3.Y, p3.Z));
    }
  }
}
Console.Error.WriteLine($"{files} ybn files; materials: " + string.Join(", ", counts.OrderByDescending(k => k.Value).Select(k => $"{k.Key}={k.Value}")));
return 0;
