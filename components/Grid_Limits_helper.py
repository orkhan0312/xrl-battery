def grid_limit(daily_import_limit_kwh: float = 50.0,
    daily_export_limit_kwh: float = 20.0,
    # fine model (choose either per_kwh_fine > 0 or fixed_daily_fine > 0)
    per_kwh_fine_eur: float = 0.50,
    fixed_daily_fine_eur: float = 0.0,
    day_boundary_tz: str = "local"):
    daily_import_accum = 0.0
    daily_export_accum = 0.0

    try:
        current_date_obj = getattr(self.microgrid, "current_date", None)
        if isinstance(current_date_obj, datetime.datetime):
            this_day = current_date_obj.date()
        elif isinstance(current_date_obj, datetime.date):
            this_day = current_date_obj
        else:
            # fall back to parsing or UTC now
            try:
                this_day = datetime.datetime.fromisoformat(str(current_date_obj)).date()
            except Exception:
                this_day = datetime.datetime.utcnow().date()
    except Exception:
        this_day = datetime.datetime.utcnow().date()

    if this_day != getattr(self, "_daily_date", None):
        # new day: optionally apply end-of-day fixed fine (if you want to bill at day-end instead of incremental)
        # reset accumulators
        self._daily_import_accum = 0.0
        self._daily_export_accum = 0.0
        self._daily_date = this_day

    # accumulate (per-step)
    self._daily_import_accum += float(imported_kwh)
    self._daily_export_accum += float(exported_kwh)

    # compute per-step excess (how much this step pushes you over the limit)
    import_excess = max(0.0, self._daily_import_accum - self.daily_import_limit_kwh)
    export_excess = max(0.0, self._daily_export_accum - self.daily_export_limit_kwh)

    # immediate incremental fine to apply this step:
    # We charge fine only for the portion that is newly in excess (so agent gets immediate feedback).
    # To do that, we compute previous accumulators:
    prev_import_acc = max(0.0, self._daily_import_accum - float(imported_kwh))
    prev_export_acc = max(0.0, self._daily_export_accum - float(exported_kwh))
    prev_import_excess = max(0.0, prev_import_acc - self.daily_import_limit_kwh)
    prev_export_excess = max(0.0, prev_export_acc - self.daily_export_limit_kwh)

    # newly exceeded kWh this step:
    new_import_excess_kwh = max(0.0, import_excess - prev_import_excess)
    new_export_excess_kwh = max(0.0, export_excess - prev_export_excess)

    # compute fines (per-step immediate)
    fine_this_step = 0.0
    if self.per_kwh_fine_eur > 0.0:
        fine_this_step += self.per_kwh_fine_eur * (new_import_excess_kwh + new_export_excess_kwh)

