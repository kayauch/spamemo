use strict;
use warnings;
use IO::Socket::INET;
use IO::Select;
use POSIX ":sys_wait_h";

my $listen_port = $ENV{LISTEN_PORT} || 18001;
my $upstream_host = $ENV{UPSTREAM_HOST} || "172.18.0.1";
my $upstream_port = $ENV{UPSTREAM_PORT} || 8001;

$SIG{CHLD} = sub { while (waitpid(-1, WNOHANG) > 0) {} };

my $server = IO::Socket::INET->new(
    LocalAddr => "0.0.0.0",
    LocalPort => $listen_port,
    Proto => "tcp",
    Listen => 32,
    ReuseAddr => 1,
) or die "listen failed: $!";

print "gateway proxy listening on $listen_port -> $upstream_host:$upstream_port
";

while (my $client = $server->accept()) {
    my $pid = fork();
    if (!defined $pid) {
        close $client;
        next;
    }
    if ($pid == 0) {
        close $server;
        my $upstream = IO::Socket::INET->new(
            PeerAddr => $upstream_host,
            PeerPort => $upstream_port,
            Proto => "tcp",
        );
        exit 1 unless $upstream;

        my $select = IO::Select->new($client, $upstream);
        while (1) {
            my @ready = $select->can_read();
            last unless @ready;
            for my $sock (@ready) {
                my $buffer = "";
                my $count = sysread($sock, $buffer, 65536);
                exit 0 unless defined $count && $count > 0;
                my $other = fileno($sock) == fileno($client) ? $upstream : $client;
                my $offset = 0;
                while ($offset < length($buffer)) {
                    my $written = syswrite(
                        $other,
                        $buffer,
                        length($buffer) - $offset,
                        $offset,
                    );
                    exit 0 unless defined $written && $written > 0;
                    $offset += $written;
                }
            }
        }
        exit 0;
    }
    close $client;
}
