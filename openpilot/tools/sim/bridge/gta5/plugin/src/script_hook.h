// Makes the game's interaction menu script (pi_menu) see the gameplay camera as rendering while the openpilot camera
// takes frames: it closes the menu on any frame after one drawn by a script camera.
#pragma once
#include <functional>
#include <string>

namespace script_hook {
// finds the native handler and pi_menu's program; logs and returns false when the game's code doesn't match
bool Install(std::function<void(const std::string &)> log);
// patches pi_menu's native table whenever its program is (re)loaded; call each tick
void Update();
// restores the native table, before this module unloads
void Uninstall();
// while on, IS_GAMEPLAY_CAM_RENDERING returns true to pi_menu
void SetOverride(bool on);
}  // namespace script_hook
