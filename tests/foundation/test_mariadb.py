"""Real repository read on the disposable database prepared by PHP migration tests."""
import os

import pytest
from voice_agent.repository import PublishedRepository


@pytest.mark.skipif(not os.getenv('M0_TEST_PORT'), reason='disposable MariaDB not selected')
def test_reader_loads_only_published_snapshot():
    import mysql.connector
    repository = PublishedRepository(lambda: mysql.connector.connect(
        host='127.0.0.1', port=int(os.environ['M0_TEST_PORT']),
        user='m0_reader', password='', connection_timeout=5))
    snapshot = repository.load('basic_voice')
    assert snapshot['workflow_version'] == 1
    assert snapshot['destination_id'] == 'agent:basic_voice'
    with pytest.raises(ValueError, match='agent_unavailable'):
        repository.load('unpublished_agent')
