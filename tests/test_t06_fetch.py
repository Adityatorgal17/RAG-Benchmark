from ragc.corpus.fetch import CorpusFetcher


def test_fetcher_reports_url_health_and_counts():
    fetcher = CorpusFetcher()
    report = fetcher.check_manifest_urls()

    assert report["threatintel"] >= 5
    assert report["vendor"] >= 4
    assert report["total_good"] >= 9
