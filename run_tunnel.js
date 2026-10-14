const localtunnel = require('localtunnel');

(async () => {
    try {
        const tunnel = await localtunnel({ port: 5000 });
        console.log('\n==================================================');
        console.log('   CARDIOPULSE AI IS LIVE & ACCESSIBLE PUBLICLY!  ');
        console.log('==================================================');
        console.log('PUBLIC URL: ' + tunnel.url);
        console.log('==================================================\n');

        tunnel.on('close', () => {
            console.log('Tunnel closed.');
        });
    } catch (err) {
        console.error('Tunnel error:', err);
    }
})();
