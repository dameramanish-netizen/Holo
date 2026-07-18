"""Holo desktop GUI (Tkinter). Windows-first, cross-platform capable."""
from __future__ import annotations
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from holo.app_state import AppController, SAMPLES_PER_ZONE
from holo.capture import MicCapture
from holo.profile import (
    ZoneActionConfiguration, list_profiles, load_profile, delete_profile, set_last_profile_id,
)
from holo.zone import DeskZone, ZoneActionKind


class HoloApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Holo")
        self.geometry("880x600")
        self.minsize(760, 520)

        self.controller = AppController()
        self.controller.on_tap_feedback = self._on_tap_feedback
        self.controller.on_level = self._on_level
        self.controller.on_action_status = self._on_action_status

        self.listening = False
        self.selected_device = None

        self._build_layout()
        self._refresh_profile_list()
        self._auto_load_profile()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    def _build_layout(self):
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        nav = ttk.Frame(container, width=160)
        nav.pack(side="left", fill="y")
        nav.pack_propagate(False)

        self.pages = {}
        self.page_container = ttk.Frame(container)
        self.page_container.pack(side="right", fill="both", expand=True)

        for name in ["Desk", "Calibration", "Actions", "Profiles", "Settings"]:
            btn = ttk.Button(nav, text=name, command=lambda n=name: self._show_page(n))
            btn.pack(fill="x", padx=8, pady=4)

        self._build_desk_page()
        self._build_calibration_page()
        self._build_actions_page()
        self._build_profiles_page()
        self._build_settings_page()

        self._show_page("Desk")

    def _show_page(self, name):
        for page in self.pages.values():
            page.pack_forget()
        self.pages[name].pack(fill="both", expand=True, padx=16, pady=16)
        if name == "Actions":
            self._refresh_actions_page()

    # ------------------------------------------------------------------
    def _build_desk_page(self):
        page = ttk.Frame(self.page_container)
        self.pages["Desk"] = page

        ttk.Label(page, text="Desk", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        self.profile_label = ttk.Label(page, text="No profile loaded — calibrate first.")
        self.profile_label.pack(anchor="w", pady=(4, 12))

        row = ttk.Frame(page)
        row.pack(anchor="w", pady=4)
        self.listen_button = ttk.Button(row, text="Start Listening", command=self._toggle_listening)
        self.listen_button.pack(side="left")
        self.desk_active_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Desk active (run actions)", variable=self.desk_active_var,
                         command=self._toggle_desk_active).pack(side="left", padx=12)

        self.level_bar = ttk.Progressbar(page, length=400, maximum=1.0)
        self.level_bar.pack(anchor="w", pady=12)

        ttk.Label(page, text="Activity").pack(anchor="w")
        self.log_box = tk.Listbox(page, height=18)
        self.log_box.pack(fill="both", expand=True, pady=(4, 0))

    def _build_calibration_page(self):
        page = ttk.Frame(self.page_container)
        self.pages["Calibration"] = page

        ttk.Label(page, text="Calibration", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(page, text=f"{SAMPLES_PER_ZONE} taps per zone, 4 zones, 40 total.").pack(anchor="w", pady=(0, 12))

        form = ttk.Frame(page)
        form.pack(anchor="w", fill="x")
        ttk.Label(form, text="Profile name").grid(row=0, column=0, sticky="w")
        self.calib_name = ttk.Entry(form, width=40)
        self.calib_name.insert(0, "My Desk")
        self.calib_name.grid(row=0, column=1, sticky="w", padx=8, pady=2)
        ttk.Label(form, text="Surface").grid(row=1, column=0, sticky="w")
        self.calib_surface = ttk.Entry(form, width=40)
        self.calib_surface.grid(row=1, column=1, sticky="w", padx=8, pady=2)
        ttk.Label(form, text="Laptop position").grid(row=2, column=0, sticky="w")
        self.calib_position = ttk.Entry(form, width=40)
        self.calib_position.grid(row=2, column=1, sticky="w", padx=8, pady=2)

        btns = ttk.Frame(page)
        btns.pack(anchor="w", pady=12)
        ttk.Button(btns, text="Begin Calibration", command=self._begin_calibration).pack(side="left")
        ttk.Button(btns, text="Undo Last Tap", command=self._undo_calibration_tap).pack(side="left", padx=6)
        ttk.Button(btns, text="Cancel", command=self._cancel_calibration).pack(side="left", padx=6)
        ttk.Button(btns, text="Check Quality", command=self._check_calibration_quality).pack(side="left", padx=6)
        ttk.Button(btns, text="Save Profile", command=self._save_calibration).pack(side="left", padx=6)

        redo_row = ttk.Frame(page)
        redo_row.pack(anchor="w", pady=(0, 8))
        ttk.Label(redo_row, text="Redo one zone:").pack(side="left")
        self.redo_zone_var = tk.StringVar()
        self.redo_zone_combo = ttk.Combobox(redo_row, textvariable=self.redo_zone_var, state="readonly",
                                             values=[z.display_name for z in DeskZone.all()], width=16)
        self.redo_zone_combo.pack(side="left", padx=6)
        ttk.Button(redo_row, text="Redo Selected Zone", command=self._redo_selected_zone).pack(side="left")

        self.calib_status = ttk.Label(page, text="Not calibrating.", font=("Segoe UI", 11, "bold"))
        self.calib_status.pack(anchor="w", pady=(8, 4))

        self.calib_progress = {}
        self.calib_accuracy_labels = {}
        grid = ttk.Frame(page)
        grid.pack(anchor="w", pady=8)
        for i, zone in enumerate(DeskZone.all()):
            lbl = ttk.Label(grid, text=f"{zone.display_name}: 0/{SAMPLES_PER_ZONE}")
            lbl.grid(row=i // 2, column=(i % 2) * 2, sticky="w", padx=(16, 4), pady=4)
            self.calib_progress[zone] = lbl
            acc_lbl = ttk.Label(grid, text="")
            acc_lbl.grid(row=i // 2, column=(i % 2) * 2 + 1, sticky="w", padx=(0, 16), pady=4)
            self.calib_accuracy_labels[zone] = acc_lbl

        ttk.Label(page, text="Tip: tap naturally around the highlighted zone; start listening on the Desk tab "
                              "first if it isn't already running. After all 40 taps, use Check Quality to see "
                              "per-zone accuracy before saving -- redo just the weak zone rather than starting over.",
                  wraplength=680).pack(anchor="w", pady=(12, 16))

        # ---- quick test-tap mode (classify-only, never dispatches actions) ---
        ttk.Separator(page).pack(fill="x", pady=(4, 12))
        ttk.Label(page, text="Test Taps", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        ttk.Label(page, text="After saving a profile, tap each corner and confirm Holo guesses the right zone "
                              "-- this never runs an action, even if Desk is active.", wraplength=680).pack(
            anchor="w", pady=(0, 8))

        test_row = ttk.Frame(page)
        test_row.pack(anchor="w", pady=4)
        ttk.Label(test_row, text="I'm about to tap:").pack(side="left")
        self.test_expected_var = tk.StringVar(value="(unspecified)")
        ttk.Combobox(test_row, textvariable=self.test_expected_var, state="readonly",
                     values=["(unspecified)"] + [z.display_name for z in DeskZone.all()], width=16).pack(
            side="left", padx=6)
        self.test_mode_button = ttk.Button(test_row, text="Start Test Taps", command=self._toggle_test_mode)
        self.test_mode_button.pack(side="left", padx=6)
        ttk.Button(test_row, text="Reset Tally", command=self._reset_test_tally).pack(side="left")

        self.test_result_label = ttk.Label(page, text="", font=("Segoe UI", 12, "bold"))
        self.test_result_label.pack(anchor="w", pady=(8, 4))
        self.test_tally_label = ttk.Label(page, text="", wraplength=680)
        self.test_tally_label.pack(anchor="w")

    def _build_actions_page(self):
        page = ttk.Frame(self.page_container)
        self.pages["Actions"] = page
        ttk.Label(page, text="Actions", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(page, text="One action per zone. Changes save immediately.").pack(anchor="w", pady=(0, 12))
        self.action_result_label = ttk.Label(page, text="", foreground="#666666", wraplength=680)
        self.action_result_label.pack(anchor="w", pady=(0, 8))

        self.action_widgets = {}
        for zone in DeskZone.all():
            frame = ttk.LabelFrame(page, text=zone.display_name)
            frame.pack(fill="x", pady=6)

            kind_var = tk.StringVar()
            kind_combo = ttk.Combobox(frame, textvariable=kind_var, state="readonly",
                                       values=[k.display_name for k in ZoneActionKind])
            kind_combo.grid(row=0, column=0, padx=6, pady=6, sticky="w")

            text_var = tk.StringVar()
            text_entry = ttk.Entry(frame, textvariable=text_var, width=40)
            text_entry.grid(row=0, column=1, padx=6, pady=6, sticky="we")

            test_btn = ttk.Button(frame, text="Test", command=lambda z=zone: self._test_action(z))
            test_btn.grid(row=0, column=2, padx=6, pady=6)
            save_btn = ttk.Button(frame, text="Save", command=lambda z=zone: self._save_action(z))
            save_btn.grid(row=0, column=3, padx=6, pady=6)

            frame.columnconfigure(1, weight=1)
            self.action_widgets[zone] = (kind_var, text_var)

    def _build_profiles_page(self):
        page = ttk.Frame(self.page_container)
        self.pages["Profiles"] = page
        ttk.Label(page, text="Profiles", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(page, text="One profile per desk/laptop position. Recalibrate if you move.").pack(
            anchor="w", pady=(0, 12))

        self.profile_list = tk.Listbox(page, height=14)
        self.profile_list.pack(fill="both", expand=True)

        btns = ttk.Frame(page)
        btns.pack(anchor="w", pady=8)
        ttk.Button(btns, text="Load", command=self._load_selected_profile).pack(side="left")
        ttk.Button(btns, text="Delete", command=self._delete_selected_profile).pack(side="left", padx=6)
        ttk.Button(btns, text="Refresh", command=self._refresh_profile_list).pack(side="left", padx=6)

    def _build_settings_page(self):
        page = ttk.Frame(self.page_container)
        self.pages["Settings"] = page
        ttk.Label(page, text="Settings", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(page, text="Input device").pack(anchor="w", pady=(12, 2))

        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(page, textvariable=self.device_var, state="readonly", width=60)
        self.device_combo.pack(anchor="w")
        ttk.Button(page, text="Refresh devices", command=self._refresh_devices).pack(anchor="w", pady=8)
        self._refresh_devices()

        ttk.Label(
            page,
            text="Holo listens on the selected input continuously while Start Listening is on. "
                 "Actions only run while Desk is active on the Desk tab. Screenshot-to-clipboard and "
                 "some actions require pywin32 on Windows.",
            wraplength=600,
        ).pack(anchor="w", pady=16)

    # ---- devices --------------------------------------------------------
    def _refresh_devices(self):
        try:
            devices = MicCapture.list_input_devices()
        except Exception as exc:
            devices = []
            messagebox.showwarning("Audio", f"Could not list input devices: {exc}")
        self._devices = devices
        self.device_combo["values"] = [f"{i}: {name}" for i, name in devices]
        if devices:
            self.device_combo.current(0)
            self.selected_device = devices[0][0]
        self.device_combo.bind("<<ComboboxSelected>>", self._on_device_selected)

    def _on_device_selected(self, _event=None):
        idx = self.device_combo.current()
        if 0 <= idx < len(self._devices):
            self.selected_device = self._devices[idx][0]

    # ---- desk -------------------------------------------------------------
    def _toggle_listening(self):
        if self.listening:
            self.controller.stop_listening()
            self.listening = False
            self.listen_button.config(text="Start Listening")
            self._log("Stopped listening.")
        else:
            try:
                self.controller.start_listening(device=self.selected_device)
            except Exception as exc:
                messagebox.showerror("Microphone", f"Could not start capture: {exc}")
                return
            self.listening = True
            self.listen_button.config(text="Stop Listening")
            self._log("Listening started.")

    def _toggle_desk_active(self):
        self.controller.set_desk_active(self.desk_active_var.get())

    def _on_level(self, rms: float):
        self.after(0, lambda: self.level_bar.config(value=min(rms * 20, 1.0)))

    def _on_tap_feedback(self, zone_or_mode, reason, quality_summary):
        def update():
            if zone_or_mode == "calibration" and self.controller.calibration:
                self._refresh_calibration_progress()
                return
            if reason:
                self._log(f"Rejected ({reason}) — {quality_summary}")
            elif zone_or_mode:
                self._log(f"Tap: {zone_or_mode} — {quality_summary}")
        self.after(0, update)

    def _on_action_status(self, msg: str):
        self.after(0, lambda: self._log(f"Action: {msg}"))

    def _log(self, msg: str):
        self.log_box.insert(0, msg)
        if self.log_box.size() > 200:
            self.log_box.delete(200, tk.END)

    # ---- calibration --------------------------------------------------
    def _begin_calibration(self):
        if not self.listening:
            if not messagebox.askyesno("Not listening", "Start listening now to begin calibration?"):
                return
            self._toggle_listening()
        self.controller.begin_calibration()
        self.calib_status.config(text=f"Armed: {DeskZone.LEFT_TOP.display_name} — {DeskZone.LEFT_TOP.instruction}")
        self._refresh_calibration_progress()

    def _undo_calibration_tap(self):
        if self.controller.calibration:
            self.controller.calibration.undo_last()
            self._refresh_calibration_progress()

    def _cancel_calibration(self):
        self.controller.cancel_calibration()
        self.calib_status.config(text="Not calibrating.")
        for lbl in self.calib_accuracy_labels.values():
            lbl.config(text="")

    def _refresh_calibration_progress(self):
        session = self.controller.calibration
        if not session:
            return
        counts = session.counts()
        for zone, lbl in self.calib_progress.items():
            lbl.config(text=f"{zone.display_name}: {counts[zone]}/{SAMPLES_PER_ZONE}")
        if session.is_complete:
            self.calib_status.config(text="All 40 taps collected. Use Check Quality, then Save Profile.")
        else:
            z = session.current_zone
            self.calib_status.config(text=f"Armed: {z.display_name} — {z.instruction}")

    def _check_calibration_quality(self):
        session = self.controller.calibration
        if not session or not session.is_complete:
            messagebox.showwarning("Calibration", "Collect all 40 taps before checking quality.")
            return
        result = self.controller.calibration_quality_breakdown()
        if result is None or result[0] is None:
            messagebox.showinfo("Calibration", "Not enough samples yet to estimate accuracy.")
            return
        overall, per_zone = result
        worst_zone, worst_acc = None, 1.1
        for zone, lbl in self.calib_accuracy_labels.items():
            acc, correct, total = per_zone[zone]
            if acc is None:
                lbl.config(text="")
                continue
            lbl.config(text=f"({acc * 100:.0f}%, {correct}/{total})",
                       foreground="#1a7f37" if acc >= 0.8 else ("#c0392b" if acc < 0.6 else "#b8860b"))
            if acc < worst_acc:
                worst_acc, worst_zone = acc, zone
        suggestion = ""
        if worst_zone is not None and worst_acc < 0.8:
            suggestion = f" Weakest: {worst_zone.display_name} ({worst_acc * 100:.0f}%) — consider redoing it."
        self.calib_status.config(
            text=f"Leave-one-out accuracy: {overall * 100:.0f}%.{suggestion}",
            foreground="#1a7f37" if overall >= 0.8 else "#c0392b",
        )

    def _redo_selected_zone(self):
        session = self.controller.calibration
        if not session:
            messagebox.showwarning("Calibration", "Begin calibration first.")
            return
        name = self.redo_zone_var.get()
        if not name:
            messagebox.showwarning("Calibration", "Pick a zone to redo first.")
            return
        zone = next(z for z in DeskZone.all() if z.display_name == name)
        session.redo_zone(zone)
        self.calib_accuracy_labels[zone].config(text="")
        self._refresh_calibration_progress()

    def _save_calibration(self):
        session = self.controller.calibration
        if not session or not session.is_complete:
            messagebox.showwarning("Calibration", "Collect all 40 taps before saving.")
            return
        try:
            profile = self.controller.save_calibration(
                self.calib_name.get().strip() or "My Desk",
                self.calib_surface.get().strip(),
                self.calib_position.get().strip(),
            )
        except Exception as exc:
            messagebox.showerror("Calibration", f"Could not train classifier: {exc}")
            return
        loo = profile.calibration.leave_one_out_accuracy
        loo_text = f"{loo * 100:.0f}%" if loo is not None else "n/a"
        self.calib_status.config(text=f"Saved '{profile.name}'. Leave-one-out accuracy: {loo_text}. "
                                       f"Use Test Taps below to sanity-check before assigning actions.",
                                  foreground="#1a7f37" if (loo or 0) >= 0.8 else "#c0392b")
        self.profile_label.config(text=f"Profile: {profile.name} (leave-one-out {loo_text})")
        self._refresh_profile_list()
        self._refresh_actions_page()

    # ---- test-tap mode (classify only, no actions) -----------------------
    def _toggle_test_mode(self):
        if self.controller.profile is None:
            messagebox.showwarning("Test Taps", "Save or load a profile first.")
            return
        if not self.controller.test_mode:
            if not self.listening:
                if not messagebox.askyesno("Not listening", "Start listening now to test taps?"):
                    return
                self._toggle_listening()
            self.controller.on_test_result = self._on_test_result
            self.controller.begin_test_mode()
            self.test_mode_button.config(text="Stop Test Taps")
            self.test_result_label.config(text="Listening for test taps...")
        else:
            self.controller.end_test_mode()
            self.test_mode_button.config(text="Start Test Taps")

    def _reset_test_tally(self):
        self.controller.test_results = []
        self.test_tally_label.config(text="")

    def _on_test_result(self, decision, expected_zone):
        def update():
            got_name = DeskZone(decision.zone).display_name if decision.zone is not None else None
            if decision.rejection_reason is not None:
                text = f"Rejected: {decision.rejection_reason.display_name}"
                color = "#c0392b"
            elif expected_zone is not None:
                correct = decision.zone == expected_zone.value
                text = f"Got {got_name} (expected {expected_zone.display_name}) — {'correct' if correct else 'WRONG'}"
                color = "#1a7f37" if correct else "#c0392b"
            else:
                text = f"Got {got_name} (confidence {decision.confidence:.2f})"
                color = "#333333"
            self.test_result_label.config(text=text, foreground=color)

            correct, total, per_zone = self.controller.test_tally()
            if total:
                breakdown = ", ".join(
                    f"{z.display_name} {c}/{t}" for z, (c, t) in per_zone.items() if t
                )
                self.test_tally_label.config(text=f"Scored: {correct}/{total} ({correct / total * 100:.0f}%)  —  {breakdown}")
        self.after(0, update)

    # ---- actions --------------------------------------------------------
    def _refresh_actions_page(self):
        profile = self.controller.profile
        for zone, (kind_var, text_var) in self.action_widgets.items():
            if profile:
                action = profile.action_for(zone)
                kind_var.set(action.kind.display_name)
                text_var.set(action.text or action.path)
            else:
                kind_var.set(ZoneActionKind.NONE.display_name)
                text_var.set("")

    def _save_action(self, zone: DeskZone) -> bool:
        if not self.controller.profile:
            messagebox.showwarning("Actions", "Load or create a profile first.")
            return False
        kind_var, text_var = self.action_widgets[zone]
        kind = next(k for k in ZoneActionKind if k.display_name == kind_var.get())
        # Guard against pasted Windows paths that include surrounding quotes
        # (e.g. Explorer's "Copy as path") -- strip them here too so the
        # saved value matches what's shown in the field.
        value = text_var.get().strip().strip('"').strip("'")
        is_path_kind = kind in (ZoneActionKind.OPEN_APPLICATION, ZoneActionKind.OPEN_ITEM, ZoneActionKind.RUN_SCRIPT)
        action = ZoneActionConfiguration(
            kind=kind,
            text="" if is_path_kind else value,
            path=value if is_path_kind else "",
        )
        self.controller.save_action(zone, action)
        self.action_result_label.config(text=f"Saved {zone.display_name}: {kind.display_name}", foreground="#666666")
        self._log(f"Saved action for {zone.display_name}: {kind.display_name}")
        return True

    def _test_action(self, zone: DeskZone):
        if not self._save_action(zone):
            return
        action = self.controller.profile.action_for(zone)
        from holo import actions as actions_module
        success, message = actions_module.run_action(action, on_status=self.on_action_status_from_gui)
        color = "#1a7f37" if success else "#c0392b"
        self.action_result_label.config(text=f"{zone.display_name}: {message}", foreground=color)
        if not success:
            messagebox.showerror("Action failed", f"{zone.display_name} ({action.kind.display_name}):\n\n{message}")

    def on_action_status_from_gui(self, msg: str):
        self._log(f"Action: {msg}")

    # ---- profiles ---------------------------------------------------------
    def _refresh_profile_list(self):
        self.profile_list.delete(0, tk.END)
        for pid in list_profiles():
            self.profile_list.insert(tk.END, pid)

    def _load_selected_profile(self):
        sel = self.profile_list.curselection()
        if not sel:
            return
        pid = self.profile_list.get(sel[0])
        try:
            profile = load_profile(pid)
        except Exception as exc:
            messagebox.showerror("Profiles", f"Could not load profile: {exc}")
            return
        self.controller.profile = profile
        set_last_profile_id(profile.id)
        loo = profile.calibration.leave_one_out_accuracy
        loo_text = f"{loo * 100:.0f}%" if loo is not None else "n/a"
        self.profile_label.config(text=f"Profile: {profile.name} (leave-one-out {loo_text})")
        self._refresh_actions_page()
        self._log(f"Loaded profile '{profile.name}'")

    def _auto_load_profile(self):
        """Restores the last-used calibration on startup, so you don't have
        to recalibrate (or even re-click Load) every time the app opens."""
        profile = self.controller.auto_load_last_profile()
        if profile is None:
            return
        loo = profile.calibration.leave_one_out_accuracy
        loo_text = f"{loo * 100:.0f}%" if loo is not None else "n/a"
        self.profile_label.config(text=f"Profile: {profile.name} (leave-one-out {loo_text})")
        self._refresh_actions_page()
        self._log(f"Restored profile '{profile.name}' from last session")

    def _delete_selected_profile(self):
        sel = self.profile_list.curselection()
        if not sel:
            return
        pid = self.profile_list.get(sel[0])
        if messagebox.askyesno("Delete profile", f"Delete profile '{pid}'? This cannot be undone."):
            delete_profile(pid)
            self._refresh_profile_list()

    # ---- lifecycle --------------------------------------------------------
    def _on_close(self):
        try:
            self.controller.stop_listening()
        except Exception:
            pass
        self.destroy()


def main():
    app = HoloApp()
    app.mainloop()


if __name__ == "__main__":
    main()
